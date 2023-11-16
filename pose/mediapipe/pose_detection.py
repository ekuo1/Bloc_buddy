# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import time
import cv2
import mediapipe as mp
import copy
from pathlib import Path
from smooth_pose import smooth_pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

name = "red_overhang"

# convert to absolute path
script_path = Path(__file__).parent # identifies path where the script is
video_location = f"../../test_examples/videos/{name}.mp4" # relative to script path
video_path = script_path / video_location
video_path = str(video_path.resolve())
output_path = script_path / f"output_{name}.mp4" # writes output to the same location as script
output_path = str(output_path.resolve())

print("[INFO] starting video file thread...")
fvs = FileVideoStream(video_path).start()
time.sleep(1.0)

# determine size of output vid
width = int(fvs.stream.get(3)) 
height = int(fvs.stream.get(4)) 
frame_width = 720
frame_height = 1280
if width > height:
	frame_width = 1280
	frame_height = 720

# start the FPS timer and define the output video writer
fps = FPS().start()
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
writer = cv2.VideoWriter(output_path, fourcc, 10, (frame_width,frame_height) )   
frame_count = 0

landmark_set_for_smoothing = []

with mp_pose.Pose(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5, model_complexity = 1) as pose:

	# loop over frames from the video file stream
	while fvs.more():
		frame = fvs.read()
		if frame is None:
			break
		
		frame_count += 1
		# perform detection on every 3 frames 
		if frame_count % 3 == 0:
			frame.flags.writeable = False
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			results = pose.process(frame)	

			frame.flags.writeable = True
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			if results.pose_landmarks:
				# copy landmark for smoothing
				landmark_set_for_smoothing.append(copy.deepcopy(results.pose_landmarks.landmark))

				# replace landmarks with the smoothed ones generated
				smooth_pose(landmark_set_for_smoothing, results.pose_landmarks)

				mp_drawing.draw_landmarks(
					frame,
					results.pose_landmarks,
					mp_pose.POSE_CONNECTIONS,
					landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
				
				'''
				# Iterate over the detected landmarks.
				# stores 33 landmarks in a list, with x, y, z, visibility values
				for i, landmark in enumerate(results.pose_landmarks.landmark):
					# Append the landmark into the list.
					# feed landmarks in this form into the next algorithm
					landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))
					# Display the found landmarks after converting them into their original scale.

					# names of landmarks are stored in a different list, so need to index them
					print(f'{mp_pose.PoseLandmark(i).name}:') 
					print(f'x: {landmark.x * width}')
					print(f'y: {landmark.y * height}')
					print(f'z: {landmark.z * width}')
					print(f'visibility: {landmark.visibility}\n')
				'''
			elif landmark_set_for_smoothing:
				landmark_set_for_smoothing.pop(0)
			
			# show the frame and update the FPS counter
			frame = cv2.resize(frame, (frame_width, frame_height))
			writer.write(frame)

			# resizing for viewing purposes only
			frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
			cv2.imshow("Frame", frame)
			cv2.waitKey(1)

		fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
fvs.stop()
# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()
writer.release()

