#started 18/04/23 by Juliette SMith
# general video imports
import time
import cv2
import copy
import os
#from datetime import timedelta
from pathlib import Path
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
from scipy.spatial import distance_matrix
from boulder_detection.ModelProcessing import findBottomEdge



# # pose detection imports
# import mediapipe as mp
# from pose.mediapipe.smooth_pose import smooth_pose, time_store

# # boulder detection imports
# from boulder_detection.RouteDetector import mask_segmentation, routeDetector, showPredefinedroute, getFirstFrame, homographyMatrix, viewImage
# import numpy as np

# #vid processing imports
# from video_processing import rolling_mat, image_mask, first_frame

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#aim for pose path tracking, redesign for rolling pathtracking

#input path_track_preprocessing, timewise keypoints & boulder coords
#output screenshots of keymoves

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# check if boulders are held
def check_overlap(image_mask,pose_mask,time_kp,height,width,hold_times_loc):
    grey_mask= cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Display', grey_mask) 
    #viewImage(image_mask) #boulders

    #cv2.imshow('Display', pose_mask) 
    #viewImage(pose_mask) 
    #hold conditions
    coord1= pose_mask>0 #value is 1 or 2
    grey_mask=grey_mask.astype(bool)
    grey_pose= grey_mask==pose_mask #pose boulder 1/0 same, not the same except for human has moves/slight diff from homography
    
    grey1= np.logical_and(coord1,grey_pose) #element wise (338,600) (338,600)
    hold_indices=np.argwhere(grey1 == True) #coordinate tuples [[0 1][4 7]...]
    if len(hold_indices)!=0: #not empty
        #print(hold_indices)
        #if len(hold_times_loc)==0:
        hold_times_loc.append(time_kp)
        #else:
            #hold_times_loc=np.concatenate((hold_times_loc,time_kp),axis=0)
        #the current frame number appended, assume one of the hands is overlapping
    return hold_times_loc


###########################
#theres an errros in this, mat size should be dynamic, bc amount of time b/w on_holds is not the same
#e.g. initial move not counted bc using hold_time_loc>6 instead of frames>6
#need to fix
##########################
#new check hold func
def check_hold(second_frame_L,hold_times_loc,dist_margin,l_counter,r_counter, lf_counter, rf_counter, LH_Move, RH_Move, LF_Move, RF_Move):
    len_mat=3
    each_time_mat_LH=np.zeros((len_mat,3)) #can replace to not hardcode
    each_time_mat_RH=np.zeros((len_mat,3))
    for i in range(len_mat): #second_frame_L): #iterate 6
        #contruct current relevant time/loc matrices
        each_time_mat_LH[i][:]=hold_times_loc[len(hold_times_loc)-1-i][0:3] #RHS [frame,x_coors,ycoord]x6 vertically stacked
        each_time_mat_RH[i][:]=np.concatenate((hold_times_loc[len(hold_times_loc)-1-i][0],hold_times_loc[len(hold_times_loc)-1-i][3:5]),axis=None)
    #check if each 6s times are continuous
    #LH
    start_t=each_time_mat_LH[-1][0]
    end_t=each_time_mat_LH[0][0]
    #trial remove time threshold, just need consecutive
    #if (abs(start_t-end_t)>=5)&(abs(start_t-end_t)<10): #set min & max time(numframe) b/w
    #need logic to compare to existing classes
    dist_mat1=distance_matrix(each_time_mat_LH[:,1:3],each_time_mat_LH[:,1:3]) #using just coords
    dist_mat1=dist_mat1+np.identity(len(dist_mat1))*dist_margin #excluding along centre of matrices)
    hold_true=(dist_mat1<=dist_margin)
    if np.all(hold_true):
        if len(LH_Move)!=0:
            k=len(LH_Move) #iterate downwards
            while k > 0: #loop dict to check already exists
                if (LH_Move[k-1]['end_t']<end_t)&(LH_Move[k-1]['end_t']>start_t):
                    #only change times, do not add new hold true frame, do not update counter
                    LH_Move[k-1]['hold_frames'].append(each_time_mat_LH[1][0])
                    LH_Move[k-1]['end_t']=end_t
                    break
                k-=1
            if k==0:
                #call add to dict 
                LH_Move=add_move_dict(each_time_mat_LH,second_frame_L,l_counter,start_t,end_t,LH_Move)
        else:
            LH_Move=add_move_dict(each_time_mat_LH,second_frame_L,l_counter,start_t,end_t,LH_Move)
    #RH
    start_t=each_time_mat_RH[-1][0]
    end_t=each_time_mat_RH[0][0]
    #if (abs(each_time_mat_RH[-1][0]-each_time_mat_RH[0][0])>=5)&(abs(each_time_mat_RH[-1][0]-each_time_mat_RH[0][0])<10): 
        #need logic to compare to existing classes
    dist_mat2=distance_matrix(each_time_mat_RH[:,1:3], each_time_mat_RH[:,1:3]) #using just coords
    dist_mat2=dist_mat2+np.identity(len(dist_mat2))*dist_margin #excluding along centre of matrices)
    hold_true=(dist_mat2<=dist_margin)
    if np.all(hold_true):
        if len(RH_Move)!=0:
            k=len(RH_Move)
            while k > 0: #loop dict to check already exists
                if (RH_Move[k-1]['end_t']<end_t)&(RH_Move[k-1]['end_t']>start_t):
                    #only change times, do not add new hold true frame, do not update counter
                    RH_Move[k-1]['hold_frames'].append(each_time_mat_RH[1][0])
                    RH_Move[k-1]['end_t']=end_t
                    break
                k-=1
            if k==0:
                #call add to dict 
                RH_Move=add_move_dict(each_time_mat_RH,second_frame_L,r_counter,start_t,end_t,RH_Move)
        else:
            RH_Move=add_move_dict(each_time_mat_RH,second_frame_L,r_counter,start_t,end_t,RH_Move)
    ##### add in feet movements
    each_time_mat_LF=np.zeros((len_mat,3)) #can replace to not hardcode
    each_time_mat_RF=np.zeros((len_mat,3))
    for i in range(len_mat): #second_frame_L): #iterate 6
        #contruct current relevant time/loc matrices
        each_time_mat_LF[i][:]=np.concatenate((hold_times_loc[len(hold_times_loc)-1-i][0],hold_times_loc[len(hold_times_loc)-1-i][5:7]),axis=None) #RHS [frame,x_coors,ycoord]x6 vertically stacked
        each_time_mat_RF[i][:]=np.concatenate((hold_times_loc[len(hold_times_loc)-1-i][0],hold_times_loc[len(hold_times_loc)-1-i][7:9]),axis=None)
    #check if each 6s times are continuous
    #LH
    start_t=each_time_mat_LF[-1][0]
    end_t=each_time_mat_LF[0][0]
    #trial remove time threshold, just need consecutive
    #if (abs(start_t-end_t)>=5)&(abs(start_t-end_t)<10): #set min & max time(numframe) b/w
    #need logic to compare to existing classes
    dist_mat1=distance_matrix(each_time_mat_LF[:,1:3],each_time_mat_LF[:,1:3]) #using just coords
    dist_mat1=dist_mat1+np.identity(len(dist_mat1))*dist_margin #excluding along centre of matrices)
    hold_true=(dist_mat1<=dist_margin)
    if np.all(hold_true):
        if len(LF_Move)!=0:
            k=len(LF_Move) #iterate downwards
            while k > 0: #loop dict to check already exists
                if (LF_Move[k-1]['end_t']<end_t)&(LF_Move[k-1]['end_t']>start_t):
                    #only change times, do not add new hold true frame, do not update counter
                    LF_Move[k-1]['hold_frames'].append(each_time_mat_LF[1][0])
                    LF_Move[k-1]['end_t']=end_t
                    break
                k-=1
            if k==0:
                #call add to dict 
                LF_Move=add_move_dict(each_time_mat_LF,second_frame_L,lf_counter,start_t,end_t,LF_Move)
        else:
            LF_Move=add_move_dict(each_time_mat_LF,second_frame_L,lf_counter,start_t,end_t,LF_Move)
    #RF
    start_t=each_time_mat_RF[-1][0]
    end_t=each_time_mat_RF[0][0]
    #if (abs(each_time_mat_RH[-1][0]-each_time_mat_RH[0][0])>=5)&(abs(each_time_mat_RH[-1][0]-each_time_mat_RH[0][0])<10): 
        #need logic to compare to existing classes
    dist_mat2=distance_matrix(each_time_mat_RF[:,1:3], each_time_mat_RF[:,1:3]) #using just coords
    dist_mat2=dist_mat2+np.identity(len(dist_mat2))*dist_margin #excluding along centre of matrices)
    hold_true=(dist_mat2<=dist_margin)
    if np.all(hold_true):
        if len(RF_Move)!=0:
            k=len(RF_Move)
            while k > 0: #loop dict to check already exists
                if (RF_Move[k-1]['end_t']<end_t)&(RF_Move[k-1]['end_t']>start_t):
                    #only change times, do not add new hold true frame, do not update counter
                    RF_Move[k-1]['hold_frames'].append(each_time_mat_RF[1][0])
                    RF_Move[k-1]['end_t']=end_t
                    break
                k-=1
            if k==0:
                #call add to dict 
                RF_Move=add_move_dict(each_time_mat_RF,second_frame_L,rf_counter,start_t,end_t,RF_Move)
        else:
            RF_Move=add_move_dict(each_time_mat_RF,second_frame_L,rf_counter,start_t,end_t,RF_Move)
    #add to library of existing continuous segments, so as not to repeat. 
    #return classes that incl each hand, num moves, start and ene time
    return l_counter,r_counter, lf_counter,rf_counter,LH_Move,RH_Move, LF_Move,RF_Move

#for above
def add_move_dict(each_time_mat_LH,second_frame_L,l_counter,start_t,end_t,LH_Move):
    l_counter += 1
    #add to dict
    counter1 = str(l_counter)
    full_name = ('Lmove'+counter1)
    temp_dict= {
        "move": full_name,
        "pos": each_time_mat_LH[1,1:3],#each_time_mat_LH[int(second_frame_L/2),1:3],
        "start_t": start_t,
        "end_t":end_t,
        "hold_frames":[each_time_mat_LH[1][0]]
        }
    LH_Move.append(temp_dict) #all len
    return LH_Move

def get_moves_pose(pose_store,comb_dict):
    #first find all moves- and associated key pose locations
    moves_pose=[]
    diff_pose_store=[]
    cont=0
    pose_store=np.array(pose_store)
    comb_dict2=np.copy(comb_dict)
    for j in range(len(comb_dict2)): #iterate all poses
        #for j in range(len(comb_dict)): #iterate all screenshote
        if comb_dict2[j] in pose_store[:,0]:
            index= np.where(pose_store[:,0] == comb_dict2[j])[0]
            if cont==0:
                cont+=1
            else:
                if len(diff_pose_store)==0:
                    diff_pose_store=np.append(diff_pose_store, abs(pose_store[index]-moves_pose[-1]), axis=None)
                    diff_pose_store=diff_pose_store.reshape(1, 9)
                else:
                    diff_pose_store=np.append(diff_pose_store, abs(pose_store[index]-moves_pose[-1]), axis=0)
                #diff_pose_store.append(abs(pose_store[index]-moves_pose[-1])) #take difference before incrementation
            moves_pose.append(pose_store[index])
        else:
            #comb_dict.remove(pose_store[i][0])
            comb_dict = np.delete(comb_dict, np.where(comb_dict == comb_dict2[j]))
    #if does not exist within at all

    
    diff_pose_store=np.delete(diff_pose_store, 0, 1) #delete frames column
    return [moves_pose,diff_pose_store,comb_dict]


def condolidate_metrics(pose_store,comb_dict,chosen_FPS,frame_interval,input_fps,min_height,max_height):
    #first find all moves- and associated key pose locations
    #update move_pose and diff_pose_store
    [moves_pose,diff_pose_store,comb_dict]=get_moves_pose(pose_store,comb_dict)
    diff_pose=np.mean(diff_pose_store)
    difficulties=[50,100,150] #to be adjusted
    

    #overall route time- this calculation is WRONG
    #need some maths to convert to seconds
    total_num_frames=comb_dict[-1]-comb_dict[0]
    moves_time=total_num_frames/input_fps #(output s) #frame_int=fps
    av_move_time=moves_time/(len(comb_dict))

    #vert perc route complete
    #from above should already know min and max boulder height
    moves_pose=np.array(moves_pose)
    moves_pose=np.concatenate( moves_pose, axis=0 )
    hand_pose_y=moves_pose[:,[2,4]] #last row, colum 3,5 hand y vals
    hand_pose_y=np.max(hand_pose_y) 
    perc_complete=(hand_pose_y-min_height)/(max_height-min_height) #line of hard coding
    if perc_complete>1:
        perc_complete=1.0

    #add measure of difficultly
    #call difficulty function

    final_climb_metrics= {
        "climb_time_taken": moves_time,
        "av_time_bw_moves": av_move_time,
        "vert_climb_compl": perc_complete * 100,
        "difficulty": 'relative_difficulty'
        }
    return final_climb_metrics

#from mask (color route)
def boulder_heights(route_dict, color_route):
    curr_boulders=route_dict[color_route] #use boulder locations as spacified in route_route dict
    curr_bould_y=[]
    curr_bould_x=[]
    curr_boulder_size=[]
    for i in range(len(curr_boulders)):
        temp=np.squeeze(curr_boulders[i])
        #find y average
        y=np.round(np.average(temp[:,1]))
        x=np.round(np.average(temp[:,0]))
        curr_bould_y.append(y)
        curr_bould_x.append(x)
        curr_boulder_size.append(len(temp))
    #min & max boulder locations on entire route
    min_height=min(curr_bould_y)
    max_height=max(curr_bould_y)

    #metric for relative boulder size
    #sum all segmented boulders of correct colour
    relative_boulder_size=np.sum(curr_boulder_size)
    return curr_bould_y, curr_bould_x, min_height, max_height, relative_boulder_size

#def vel_method():
def vel_method(time_kp_prev,time_kp,velocities):
    #velocity=dis/time
    LH_vel=(time_kp[1:3]-time_kp_prev[1:3])/(time_kp[0]-time_kp_prev[0])
    RH_vel=(time_kp[3:5]-time_kp_prev[3:5])/(time_kp[0]-time_kp_prev[0])
    LF_vel=(time_kp[5:7]-time_kp_prev[5:7])/(time_kp[0]-time_kp_prev[0])
    RF_vel=(time_kp[7:9]-time_kp_prev[7:9])/(time_kp[0]-time_kp_prev[0])
    temp=np.array(np.concatenate((time_kp[0],LH_vel,RH_vel,LF_vel,RF_vel), axis=None))
    velocities.append(temp)
    return velocities #vel=[nx9]

#from velocities find cosine similarity (post process find sig frames)
from numpy.linalg import norm
#def cosine_sig_frames(velocities,step):
def cosine_sig_frames(velocities):
    cosine_L=[]
    cosine_R=[]
    #find cosine similarity
    for i in range(1,len(velocities)): #start at 2nd index
        L_prev=velocities[i-1][1:3]
        L_curr=velocities[i][1:3]
        R_prev=velocities[i-1][3:5]
        R_curr=velocities[i][3:5]
        temp=cos_sim(L_curr, L_prev) #call cosine similarity fucntion
        cosine_L.append(temp)
        temp=cos_sim(R_curr, R_prev) #call cosine similarity fucntion
        cosine_R.append(temp)
    cosine_L= np.array(cosine_L)
    cosine_R= np.array(cosine_R)
    #convert cosine NaN to v small values
    where_are_NaNs = np.isnan(cosine_L)
    cosine_L[where_are_NaNs] = 10**-9
    where_are_NaNs = np.isnan(cosine_R)
    cosine_R[where_are_NaNs] = 10**-9
    #filter
    L_filtered=savgol_smoothing(cosine_L)
    R_filtered=savgol_smoothing(cosine_R)
    #find troughs- no movements
    [L_min,L_max]=find_troughs(L_filtered) #want min similarity to show movement
    [R_min,R_max]=find_troughs(R_filtered)

    return cosine_L,cosine_R,L_filtered, R_filtered, L_max, R_max

from scipy.signal import savgol_filter
def savgol_smoothing(data):
    #perform smoothing- savgol filter (in built in scipy)
    np.set_printoptions(precision=2)  # For compact display.
    filtered_data=savgol_filter(data, 5, 2) #try 5, 11, 21, 31 window sizes for smoother filter
    return filtered_data

def cos_sim(a, b): 
    #function removes neg values in cos sim
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    minx = -1 
    maxx = 1
    return ((dot_product / (norm_a * norm_b))-minx)/(maxx-minx)

from scipy.signal import find_peaks
def find_troughs(data,v_frames):
    #min = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1         # local min
    #max = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1         # local max
    distance=round(len(data)/15)
    #pad data if unsuitable for dist=40 divisor
    if len(data)%distance !=0:
        data2=pad_data(data,distance)
        sampling=v_frames[1]-v_frames[0]
        temp= np.arange(v_frames[-1], v_frames[-1]+(len(data2)-len(v_frames))*sampling, sampling)
        v_frames2= np.concatenate((v_frames,temp),axis=None)
    else:
        data2=data
        v_frames2=v_frames
    max, _ = find_peaks(data2, distance=distance) 
    min, _= find_peaks(-data2, distance=distance)
    # difference between peaks is >= 150, distance is tunable param
    return min, max, data2 ,v_frames2

#func round to correct sampling
def round_sampling(num,frame_interval):
    return frame_interval*round(num/frame_interval)-1 #-1 for start 0 indexing

## add in feet moves
import imageio
def get_screenshots(LH_Move,RH_Move, LF_Move, RF_Move,L_indices, R_indices, Lf_indices, Rf_indices,frame_interval): 
    comb_dict=[]
    ########### for check overlap method
    #check which one longer
    if len(LH_Move)>len(RH_Move):
        long_moves=LH_Move
        short_moves=RH_Move
    else:
        long_moves=RH_Move
        short_moves=LH_Move
    #iterate move dictionaries
    #add midpoint long moves
    for i in range(len(long_moves)):
        mid_pt=int((long_moves[i]['end_t']-long_moves[i]['start_t'])/2)
        comb_dict.append(round_sampling(long_moves[i]['start_t']+mid_pt,frame_interval))
    for i in range(len(short_moves)):
        for j in range(len(long_moves)-1,-1,-1): #for i in range(N - 1, -1, -1): iterate down, check last moves first
            #check long moves similarity
            contained = [a in short_moves[i]['hold_frames'] for a in long_moves[j]['hold_frames']]
            if np.count_nonzero(contained)/len(contained)>0.8: #more than 80% similarity
                # do nothing exit for loop, confirmed same hold
                #otherwise continue to increment
                break
        if j ==0: #did not exit for loop early, iterates to 0
            #issue with midpt calc, ends up not discretised same as sampling rate
            mid_pt=int((short_moves[i]['end_t']-short_moves[i]['start_t'])/2)
            comb_dict.append(round_sampling(short_moves[i]['start_t']+mid_pt,frame_interval))
    #repeat for feet moves
    if len(LF_Move)>len(RF_Move):
        long_moves=LF_Move
        short_moves=RF_Move
    else:
        long_moves=RF_Move
        short_moves=LF_Move
    #add midpoint long moves
    for i in range(len(long_moves)):
        mid_pt=int((long_moves[i]['end_t']-long_moves[i]['start_t'])/2)
        comb_dict.append(round_sampling(long_moves[i]['start_t']+mid_pt,frame_interval))
    for i in range(len(short_moves)):
        for j in range(len(long_moves)-1,-1,-1): #for i in range(N - 1, -1, -1): iterate down, check last moves first
            #check long moves similarity
            contained = [a in short_moves[i]['hold_frames'] for a in long_moves[j]['hold_frames']]
            if np.count_nonzero(contained)/len(contained)>0.8: #more than 80% similarity
                #abs(long_moves[i]['start_t']-short_moves[j]['start_t'])<10 and abs(long_moves[i]['end_t']-short_moves[j]['end_t'])<10 :
                # do nothing exit for loop, confirmed same hold
                #otherwise continue to increment
                break
        if j ==0: #did not exit for loop early, iterates to 0
            mid_pt=int((short_moves[i]['end_t']-short_moves[i]['start_t'])/2)
            comb_dict.append(round_sampling(short_moves[i]['start_t']+mid_pt,frame_interval))
    #############for vel method
    temp=np.concatenate((L_indices, R_indices,Lf_indices, Rf_indices), axis=0)
    #temp=temp[:, np.newaxis]
    comb_dict=np.concatenate((comb_dict,temp), axis=0)
    #time_kp(comb_dict)
    ####### try to reduce
    comb_dict=list(set(comb_dict)) #remove duplicates
    comb_dict=np.sort(comb_dict) #order t-sequence
    return comb_dict

def produce_screenshots(comb_dict,reader,screenshot_path,file_list,pose_store):
    [_,_,comb_dict]=get_moves_pose(pose_store,comb_dict)
    for frame_number, im in enumerate(reader):
        # im is numpy array
        for j in range(len(comb_dict)):
            if frame_number == comb_dict[j]:
                imageio.imwrite(screenshot_path/f'frame{frame_number}.jpg', im[:,:,0:3])
                file_list.append(frame_number)
                break
    return file_list

import matplotlib.pyplot as plt
def graph_ang_v(velocities):
    ####find velcocity method sig frames
    ### graph velocities
    velocities=np.array(velocities)
    velocities=velocities[1:len(velocities)] #rid first entry
    v_frames=velocities[:,0]
    L_x_v=velocities[:,1]
    L_y_v=velocities[:,2]
    R_x_v=velocities[:,3]
    R_y_v=velocities[:,4]
    Lf_x_v=velocities[:,5]
    Lf_y_v=velocities[:,6]
    Rf_x_v=velocities[:,7]
    Rf_y_v=velocities[:,8]
    L_angular=np.sqrt(L_x_v**2+L_y_v**2) #final angular velocities
    R_angular=np.sqrt(R_x_v**2+R_y_v**2)
    Lf_angular=np.sqrt(Lf_x_v**2+Lf_y_v**2)
    Rf_angular=np.sqrt(Rf_x_v**2+Rf_y_v**2)
    #call fucntion accel method
    #call fucntion smooth velocites and acceleration
    L_angular=savgol_smoothing(L_angular)
    R_angular=savgol_smoothing(R_angular)
    Lf_angular=savgol_smoothing(Lf_angular)
    Rf_angular=savgol_smoothing(Rf_angular)
    [L_min,L_max,L_angular,v_frames2]=find_troughs(L_angular,v_frames)
    [R_min,R_max,R_angular,v_frames2]=find_troughs(R_angular,v_frames)
    [Lf_min,Lf_max,Lf_angular,v_frames2]=find_troughs(Lf_angular,v_frames)
    [Rf_min,Rf_max,Rf_angular,v_frames2]=find_troughs(Rf_angular,v_frames)
    plt.figure()
    plt.plot(v_frames2,L_angular)
    plt.plot(v_frames2,R_angular)
    plt.plot(v_frames2,Lf_angular)
    plt.plot(v_frames2,Rf_angular)
    plt.plot(v_frames2[L_min], L_angular[L_min], "o", label="min", color='r') #use min for zero velocity
    plt.plot(v_frames2[R_min], R_angular[R_min], "o", label="min", color='b')
    plt.plot(v_frames2[Lf_min], Lf_angular[Lf_min], "o", label="min", color='m')
    plt.plot(v_frames2[Rf_min], Rf_angular[Rf_min], "o", label="min", color='k')
    return v_frames, L_min, R_min, Lf_min, Rf_min

def pad_data(data,distance):
    extra=len(data)%distance
    temp=np.ones((1,extra))*100 #very large number
    data2=np.concatenate((data,temp),axis=None)
    return data2

def graph_cos_sim(v_frames,L_filtered,R_filtered,L_indices,R_indices):
    pass
    #graph cosine similiarities
    # plt.figure()
    # plt.plot(v_frames,L_filtered)
    # plt.plot(v_frames,R_filtered)
    # plt.plot(v_frames[L_indices], L_filtered[L_indices], "o", label="min", color='r') #use min for least similarity, most movement
    # plt.plot(v_frames[R_indices], R_filtered[R_indices], "o", label="min", color='b')

#remove im duplicates
def mse(imA,imB):
    #images have same dimension
    err=np.sum((imA.astype("float")-imB.astype("float"))**2)
    err/=float(imA.shape[0]*imA.shape[1])
    #lower mse=more similar
    return err

from pathlib import PureWindowsPath
def delete_duplicate(filter_thres,file_list,screenshot_path,comb_dict,contours_dict,pose_store, climb_id, gui, app_context=None):
    if gui:
        import sys
        # Find path of file name relative to the running script
        script_path = Path(__file__).parent # identifies path where the script is

        # append GUI folder to the path
        gui_path = script_path / "GUI"
        gui_path = str(gui_path.resolve())
        sys.path.append(gui_path)

        # Import the flask app and its modules
        from app import app, db
        from app.models import Climb, Screenshot

    # iterate over files in
    # that directory
    #images = Path(screenshot_path).glob('*.jpg')
    mse_err=[]
    start_frames_cond=[]
    cont=0

    screenshot_number = 1

    #get keypoints
    [move_pose,_,comb_dict]=get_moves_pose(pose_store,comb_dict)
    move_pose=np.concatenate( move_pose, axis=0 )
    move_pose=np.delete(move_pose, 0, axis=1)  #delete frames column
    
    for i in range(len(file_list)):
        im_path=str(PureWindowsPath(screenshot_path))
        if cont==0:
            im_path=im_path+'\\'+f'frame{file_list[0]}'+'.jpg' #i=0
            image= cv2.imread(im_path)
            image_prev=np.copy(image)
            #call new func to check for if start condition passed
            check_screenshots(start_frames_cond, image, contours_dict,move_pose[i]) #appends to start_frames_cond
            cont+=1
        else:
            im_path=im_path+'\\'+f'frame{file_list[i]}'+'.jpg' #next file_list iter
            image= cv2.imread(im_path)
            err=mse(image_prev,image)
            image_prev=image
            mse_err.append(err)
            #call new func to check for if start condition passed
            check_screenshots(start_frames_cond, image, contours_dict,move_pose[i]) #appends to start_frames_cond
        del im_path #reset


    #set threshold err=1500
    for i in range(len(file_list)):
        if i != len(file_list)-1 and mse_err[i]<=filter_thres: #too similar (mse has 1 shorter length)
            # delete these screenshots
            im_path=str(PureWindowsPath(screenshot_path))
            im_path=im_path+'\\'+f'frame{file_list[i]}'+'.jpg'
            os.remove(im_path)
            #remove from comb dict
            comb_dict= np.delete(comb_dict, np.where(comb_dict == file_list[i]))
        elif start_frames_cond[i]: #true
            # delete these screenshots
            im_path=str(PureWindowsPath(screenshot_path))
            im_path=im_path+'\\'+f'frame{file_list[i]}'+'.jpg'
            os.remove(im_path)
            #remove from comb dict
            comb_dict= np.delete(comb_dict, np.where(comb_dict == file_list[i]))
        else:   
            # keep these screenshots
            # change the name
            im_path=str(PureWindowsPath(screenshot_path))
            src_im_path=im_path+'\\'+f'frame{file_list[i]}'+'.jpg'
            dest_im_path=im_path+'\\'+f"{screenshot_number}.jpg"
            os.replace(src_im_path, dest_im_path)

            if gui:
                # add to the db
                with app_context:
                    screenshot = Screenshot(order=screenshot_number, seconds_elapsed=1, climb_id=climb_id)
                    db.session.add(screenshot)
                    db.session.commit()

            screenshot_number += 1

    return comb_dict #reduced list

def check_screenshots(start_frames_cond, image, contours_dict, diff_pose_store):
    #delete screenshots from list that do not fail lower edge req
    row_num=findBottomEdge(image, contours_dict)
    if (diff_pose_store[1]>=row_num) and (diff_pose_store[3]>=row_num): #all hand- y values above line
        start_frames_cond.append(False) #do not need to delete
    else:
        start_frames_cond.append(True)
    #return --> write directly through append
             
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Testing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#test each fucntion and their integration seperatly


def viewImage(image):
    cv2.imshow('Display', image)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
