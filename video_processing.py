# Last edited 24/4/23 by Emily Kuo
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Imports 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# general video imports
import time
import cv2
import copy
from pathlib import Path, PureWindowsPath
from imutils.video import FileVideoStream
from imutils.video import FPS
import imageio #for getting screenshots
import os
import sys
import matplotlib.pyplot as plt

# pose detection imports
import mediapipe as mp
from pose.mediapipe.smooth_pose import smooth_pose, time_store

# boulder detection imports
from boulder_detection.RouteDetector import mask_segmentation, routeDetector, showPredefinedroute, getFirstFrame, homographyMatrix, is_good_homography, test_orietation_homography, test_scale_homography, homography_different
from boulder_detection.ModelProcessing import getMaskFromModel, maskPostProcessing,findBottomEdge
import numpy as np
from openvino.runtime import Core, serialize

# pathtracking imports
from path_tracking_2 import check_overlap, check_hold, get_screenshots, vel_method, cosine_sig_frames, savgol_smoothing, find_troughs, produce_screenshots, graph_ang_v, graph_cos_sim, delete_duplicate, boulder_heights,condolidate_metrics, mse
from scipy.spatial import distance_matrix

# Find path of file name relative to the running script
script_path = Path(__file__).parent # identifies path where the script is

def setup():
    path = f"{script_path}/boulder_detection/model-IR.xml"
    ie = Core()
    model_ir = ie.read_model(model=path)
    model_ir.reshape([-1, 256, 256, 3]) #dynamic first parameter
    compiled_model_ir = ie.compile_model(model=model_ir, device_name="AUTO", config={"PERFORMANCE_HINT":"LATENCY"})

    # Setup for pose detection
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5, model_complexity = 1)

    return compiled_model_ir, mp_drawing, mp_drawing_styles, mp_pose, pose

def process_video(name, compiled_model_ir, mp_drawing, mp_drawing_styles, mp_pose, pose, color_route="", climb_id=-1, app_context=None):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Setup
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    gui = False
    if climb_id != -1:
        gui = True  # if run through the GUI

    if gui:
        # Run from GUI

        # append GUI folder to the path
        gui_path = script_path / "GUI"
        gui_path = str(gui_path.resolve())
        sys.path.append(gui_path)

        # Import the flask app and its modules
        from app import app, db
        from app.models import Climb, Screenshot
        #app.app_context().push()
        with app_context:
            climb = Climb.query.filter_by(id=climb_id).first()
            color_route = climb.colour
        video_location = f"GUI/instance/videos/{name}" # relative to script path
        #output_location = f"GUI/instance/climb{climb_id}/output_{name}" # relative to script path
        screenshot_path = script_path/f"GUI/instance/climb{climb_id}" #output file named same as boulder route
    else:
        climb_id = 0
        #video_location = f"test_examples/{name}.gif" # relative to script path
        video_location = f"test_examples/{name}.mp4" # relative to script path
        output_location = f"output/output_{name}.mp4"
        screenshot_path = script_path/f"output_screenshots/{name}" #output file named same as boulder route
        output_path = script_path / output_location # output file location 
        output_path = str(output_path.resolve())

    video_path = script_path / video_location
    video_path = str(video_path.resolve())

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # setup for screenshot output
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not os.path.exists(screenshot_path):
        os.makedirs(screenshot_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("[INFO] starting video file thread...")
    fvs = FileVideoStream(video_path).start()
    time.sleep(1.0)

    # Determine size of vid
    width = int(fvs.stream.get(3)) 
    height = int(fvs.stream.get(4)) 

    # Define the output video writer and chosen FPS
    input_fps = fvs.stream.get(cv2.CAP_PROP_FPS)
    chosen_FPS = 10
    # define number of frames requires for 1s & required rolling matrices
    frame_interval = int(round(input_fps / chosen_FPS)) #not sure if this is correct
    second_frame_L=int(60/chosen_FPS)

    landmark_set_for_smoothing = []
    if not gui: 
        # for size of output vid
        frame_width = 720
        frame_height = 1280
        if width > height:
            frame_width = 1280
            frame_height = 720
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter(output_path, fourcc, chosen_FPS, (frame_width,frame_height) ) 
  
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Boulder Detection on the First Frame
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    frame_count = 0
    # Start FPS timer
    fps = FPS().start()
    
    first_frame = getFirstFrame(video_path)

    if gui:
        # write first_frame to db
        with app_context:
            cv2.imwrite(f"{screenshot_path}/0.jpg", first_frame)
            screenshot = Screenshot(order=0, seconds_elapsed=1, climb_id=climb_id)
            db.session.add(screenshot)
            db.session.commit()

    #image_mask = mask_segmentation(first_frame, color_route) #finding the red route
    image_mask_pre = getMaskFromModel(first_frame, script_path, compiled_model_ir)
    image_mask = maskPostProcessing(image_mask_pre)
    image_mask_og=np.copy(image_mask)
    route_dict = routeDetector(first_frame, image_mask) #dictionary containing location of contours

    #find max height from image mask
    #later fix for min height based on start boulder?
    [curr_bould_y, curr_bould_x, min_height, max_height, relative_boulder_size]=boulder_heights(route_dict, color_route)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over Video Frames 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #define hold time
    hold_times_loc=[]
    velocities=[]
    l_counter=0
    r_counter=0
    lf_counter=0
    rf_counter=0
    time_kp_prev=[1,width,height,width,height,width,height,width,height] #set to max, avoid troughs, prev used frame_height, frame_width
    pose_store=[] #all keypoint positions
    COM_store=[]

    #dictionary of moves
    LH_Move = []
    RH_Move = []
    LF_Move = []
    RF_Move = []

    while fvs.more():
        
        # Read frame here
        frame = fvs.read()
        if frame is None:
            break

        frame_count += 1

        # Run the algorithm on every couple of frames 
        if frame_count % frame_interval == 0: #was 1

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Homography 
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            frame_previous = frame.copy() #saving frame before homography transform

            if frame_count == frame_interval: ##incase the first frame has a bad homography matrix
                #print("First frame")
                frame, Hcv = homographyMatrix(first_frame, frame)
                Hcv_previous_frame = Hcv.copy()
                homography_change = True 
                counter = 0

            if (homography_change == True) | (counter > 4): #homogrpahy matrix has changed 
                frame, Hcv = homographyMatrix(first_frame, frame)
                homography_change = homography_different(Hcv, Hcv_previous_frame)
                is_good_or_bad = is_good_homography(Hcv, first_frame.shape[1], first_frame.shape[0], 0.20) #is this homography matrix good

                if is_good_or_bad == False: #if homography matrix is bad use the previous
                    #print("Frame " + str(frame_count) + "has a bad homography transform :( ")
                    Hcv = Hcv_previous_frame
                    frame = cv2.warpPerspective(frame_previous, Hcv_previous_frame, (first_frame.shape[1],first_frame.shape[0]))

                Hcv_previous_frame = Hcv.copy()
                counter = 0

            else: 
                frame = cv2.warpPerspective(frame_previous, Hcv_previous_frame, (first_frame.shape[1],first_frame.shape[0]))
                counter += 1


            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Pose Detection 
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # Processing frame
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)	
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                # Replace pose landmarks with a smoothed version
                landmark_set_for_smoothing.append(copy.deepcopy(results.pose_landmarks.landmark))
                smooth_pose(landmark_set_for_smoothing, results.pose_landmarks)

                # Draw pose landmarks 
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                #save into time based landmark/keypoint array
                [time_kp,pose_mask,pose_store,COM_store]=time_store(fps,results,width,height,pose_store,COM_store)
                #calculate angular velocity
                velocities=vel_method(time_kp_prev,time_kp,velocities)
                time_kp_prev=time_kp #save prev entry

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #  save and call path tracking algorithm
                dist_margin=15 #arbitrary
                hold_times_loc=check_overlap(image_mask_og,pose_mask,time_kp,height,width,hold_times_loc) #hold time updated
                
                if len(hold_times_loc)>0:
                    if (len(hold_times_loc)>=second_frame_L)& (hold_times_loc[-1][0]==frame_count-1): #hold_times_loc.shape[1]>=6
                        #new func: check_movement
                        [l_counter,r_counter, lf_counter,rf_counter,LH_Move,RH_Move, LF_Move,RF_Move]=check_hold(second_frame_L,hold_times_loc,dist_margin,l_counter,r_counter, lf_counter, rf_counter, LH_Move, RH_Move, LF_Move, RF_Move)
            
            # if no pose landmarks found
            elif landmark_set_for_smoothing:
                landmark_set_for_smoothing.pop(0)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #  Processing frame for writing to output video 
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if not gui:
                # Draw boulder contours 
                for contour in route_dict[color_route]:
                    cv2.drawContours(frame, [contour],-1, (0,255,0),2, maxLevel=2)

                # Resize the frame for writing to the output video
                frame = cv2.resize(frame, (frame_width, frame_height))
                writer.write(frame)

                # Resizing the frame again for viewing on a screen (optional)
                frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
                cv2.imshow("Frame", frame)

                cv2.waitKey(1)
        fps.update()

    #graphing vel method
    [v_frames, L_max, R_max, Lf_max, Rf_max]=graph_ang_v(velocities)
    #not always producing in dicretised frames that match sampling rate

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # post processing- get screenshots & metrics
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    reader = imageio.get_reader(video_path)
    #use 2 methods for more accuracy
    #row_num= findBottomEdge(image, contours_dict) <-- call in get screenshots
    comb_dict= get_screenshots(LH_Move,RH_Move,LF_Move, RF_Move,v_frames[L_max], v_frames[R_max], v_frames[Lf_max], v_frames[Rf_max],frame_interval)
    #run pose detection on existing screenshots
    #run reduce num screenshots function
    file_list=[]
    file_list=produce_screenshots(comb_dict,reader,screenshot_path,file_list,pose_store)

    #reduce number of duplicate screenshots 
    filter_thres=1000
    comb_dict=delete_duplicate(filter_thres,file_list,screenshot_path,comb_dict,route_dict,pose_store, climb_id, gui, app_context) #was contours_dict

    #Metric: difficultly of climb
    #draw line bw first and last boulder, see how each move distance affects?
    curr_bould_x=np.expand_dims(curr_bould_x, axis=1)
    curr_bould_y=np.expand_dims(curr_bould_y, axis=1)
    boulders=np.concatenate((curr_bould_x, curr_bould_y), axis=1)     #from above
    last_boulder= boulders[np.argmax(curr_bould_y)]#need x and y coords
    first_boulder= boulders[np.argmin(curr_bould_y)]
    direct_climb_vector= last_boulder-first_boulder
    #distances from COM to vector
    COM_deviations=[]
    for i in range(len(COM_store)):
        d = np.linalg.norm(np.cross(direct_climb_vector, last_boulder-COM_store[i]))/np.linalg.norm(direct_climb_vector)
        COM_deviations.append(d)
    mean_deviation=np.mean(COM_deviations)

    #insert function for adding metrics
    final_climb_metrics=condolidate_metrics(pose_store,comb_dict,chosen_FPS,frame_interval,input_fps,min_height,max_height)

    if gui:
        with app_context:
            climb = Climb.query.filter_by(id=climb_id).first()
            climb.time_taken = final_climb_metrics["climb_time_taken"]
            climb.av_time_bw_moves = final_climb_metrics["av_time_bw_moves"] 
            climb.percent_completed = final_climb_metrics["vert_climb_compl"]
            climb.difficulty = final_climb_metrics["difficulty"]
            db.session.commit()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cleanup 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Stop FPS timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # Do a bit of cleanup
    if not gui:
        cv2.destroyAllWindows()
        writer.release()
    fvs.stop()

if __name__ == "__main__":
    """ This is executed when run from the command line """

    # Input video file name
    name = "pink_edge" #green6, green6_unfinished
    color_route = 'pink' #change distance to 30

    # Load models
    compiled_model_ir, mp_drawing, mp_drawing_styles, mp_pose, pose = setup()

    # record start time
    start = time.time()
    args = sys.argv

    process_video(name, compiled_model_ir, mp_drawing, mp_drawing_styles, mp_pose, pose, color_route=color_route)

    # record end time
    end = time.time()
    print("The time of execution of above program is :",(end-start) * 10**3, "ms")
 
