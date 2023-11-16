import numpy as np
import cv2

def average_filter(set):
    if len(set) == 8:
        set.sort()
        # get middle 2 values, ignoring the min and max
        set = set[2:-2]
    # return the average
    return sum(set)/len(set)

def smooth_pose(pose_set, results):
    # repeats for each keypoint
    for i, landmark in enumerate(results.landmark):
        x_set = [pose[i].x for pose in pose_set] 
        y_set = [pose[i].y for pose in pose_set] 
        z_set = [pose[i].z for pose in pose_set] 
        visibility_set = [pose[i].visibility for pose in pose_set] 

        x_avg = average_filter(x_set)
        y_avg = average_filter(y_set)
        z_avg = average_filter(z_set)
        visibility_avg = average_filter(visibility_set)

        # write smoothed pose to results
        landmark.x = x_avg
        landmark.y = y_avg
        landmark.z = z_avg
        landmark.visibility = visibility_avg
    
    if len(pose_set) == 8:
        # remove the oldest element of pose_set 
        pose_set.pop(0)
    
from mediapipe.framework.formats import landmark_pb2

def time_store(fps,results,width,height,pose_store, COM_store):
### additional code for time based storage
    smoothed_pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks.landmark else np.zeros(468*4)
    smoothed_pose= np.reshape(smoothed_pose,(-1,4))

    ###################
    #attempt create pose_mask inside time_store
    ####################
    #pose mask before smooth pose manipulation
    pose_mask=np.zeros((height,width)) #frame is always the same size bc homographies #emily used height, width
    hand_size=20
    hand=np.ones((hand_size,hand_size))
    #array of hand landmarks
    hand_landmarks=[[int(smoothed_pose[15][0]*width), int(smoothed_pose[15][1]*height)],
                    [int(smoothed_pose[19][0]*width), int(smoothed_pose[19][1]*height)],
                    [int(smoothed_pose[17][0]*width), int(smoothed_pose[17][1]*height)],
                    [int(smoothed_pose[21][0]*width), int(smoothed_pose[21][1]*height)],
                    [int(smoothed_pose[16][0]*width), int(smoothed_pose[16][1]*height)],
                    [int(smoothed_pose[20][0]*width), int(smoothed_pose[20][1]*height)],
                    [int(smoothed_pose[18][0]*width), int(smoothed_pose[18][1]*height)],
                    [int(smoothed_pose[22][0]*width), int(smoothed_pose[22][1]*height)]]
    #array of feet landmarks -31, 29, 27, 32, 30, 28
    feet_landmarks=[[int(smoothed_pose[31][0]*width), int(smoothed_pose[31][1]*height)],
                    [int(smoothed_pose[29][0]*width), int(smoothed_pose[29][1]*height)],
                    [int(smoothed_pose[27][0]*width), int(smoothed_pose[27][1]*height)],
                    [int(smoothed_pose[32][0]*width), int(smoothed_pose[32][1]*height)],
                    [int(smoothed_pose[30][0]*width), int(smoothed_pose[30][1]*height)],
                    [int(smoothed_pose[28][0]*width), int(smoothed_pose[28][1]*height)],]
    #fill 1s into pose_mask
    for k in range(len(hand_landmarks)):
        for i in range(0, hand.shape[0]): 
            for j in range(0, hand.shape[1]):
                if (int(hand_landmarks[k][0])-i<height) & (int(hand_landmarks[k][1])-j<width) :     #crop pose_mask to be within boundaries 
                    pose_mask[int(hand_landmarks[k][0])-i][int(hand_landmarks[k][1])-j]=pose_mask[int(hand_landmarks[k][0])-i][int(hand_landmarks[k][1])-j] + hand[i][j] 
    for k in range(len(feet_landmarks)):
        for i in range(0, hand.shape[0]): 
            for j in range(0, hand.shape[1]): 
                if (int(hand_landmarks[k][0])-i<height) & (int(hand_landmarks[k][1])-j<width) :
                    pose_mask[int(hand_landmarks[k][0])-i][int(hand_landmarks[k][1])-j]=pose_mask[int(hand_landmarks[k][0])-i][int(hand_landmarks[k][1])-j] + hand[i][j] 
    #cv2.imshow('Display', pose_mask) 
    
    LH=[] #hands
    RH=[]
    LF=[] #feet
    RF=[]
    #trial change smoothed pose (which start from diff pos)
    smoothed_pose[:,1]=1-smoothed_pose[:,1] #just y chnaged
    #L/R wrist index pinky thumb- 15, 19, 17, 21, 16, 20, 18, 22
    #LH first
    if smoothed_pose[19][3]>= 0.85: #per goals stated
        LH=[fps._numFrames, int((smoothed_pose[19][0])*width), int(smoothed_pose[19][1]*height)] #ignoring z-coords
    elif smoothed_pose[17][3]>= 0.85: #per goals stated
        LH=[fps._numFrames,int((smoothed_pose[17][0])*width), int(smoothed_pose[17][1]*height)] #was width, height, x,y
    elif smoothed_pose[21][3]>= 0.85: #per goals stated
        LH=[fps._numFrames,int((smoothed_pose[21][0])*width), int(smoothed_pose[21][1]*height)]
    elif smoothed_pose[15][3]>= 0.85: #per goals stated
        LH=[fps._numFrames,int((smoothed_pose[15][0])*width), int(smoothed_pose[15][1]*height)]
    else:
        LH=[fps._numFrames,int(smoothed_pose[19][0]*width), int(smoothed_pose[19][1]*height)]
    #RH
    if smoothed_pose[20][3]>= 0.85: #per goals stated
        RH=[int(smoothed_pose[20][0]*width), int(smoothed_pose[20][1]*height)]
    elif smoothed_pose[18][3]>= 0.85: #per goals stated
        RH=[int(smoothed_pose[18][0]*width), int(smoothed_pose[18][1]*height)]
    elif smoothed_pose[21][3]>= 0.85: #per goals stated
        RH=[int(smoothed_pose[22][0]*width), int(smoothed_pose[22][1]*height)]
    elif smoothed_pose[16][3]>= 0.85: #per goals stated
        RH=[int(smoothed_pose[16][0]*width), int(smoothed_pose[16][1]*height)]
    else:
        RH=[int(smoothed_pose[20][0]*width), int(smoothed_pose[20][1]*height)]
    #L/R toes, heel, ankle- 31, 29, 27, 32, 30, 28
    #LF first
    if smoothed_pose[31][3]>= 0.85: #per goals stated
        LF=[int(smoothed_pose[31][0]*width), int(smoothed_pose[31][1]*height)] #ignoring z-coords
    elif smoothed_pose[29][3]>= 0.85: #per goals stated
        LF=[int(smoothed_pose[29][0]*width), int(smoothed_pose[29][1]*height)] #was width, height
    elif smoothed_pose[27][3]>= 0.85: #per goals stated
        LF=[int(smoothed_pose[27][0]*width), int(smoothed_pose[27][1]*height)]
    else:
        LF=[int(smoothed_pose[31][0]*width), int(smoothed_pose[31][1]*height)]
    #RF
    if smoothed_pose[32][3]>= 0.85: #per goals stated
        RF=[int(smoothed_pose[32][0]*width), int(smoothed_pose[32][1]*height)]
    elif smoothed_pose[30][3]>= 0.85: #per goals stated
        RF=[int(smoothed_pose[30][0]*width), int(smoothed_pose[30][1]*height)]
    elif smoothed_pose[28][3]>= 0.85: #per goals stated
        RF=[int(smoothed_pose[28][0]*width), int(smoothed_pose[28][1]*height)]
    else:
        RF=[int(smoothed_pose[32][0]*width), int(smoothed_pose[32][1]*height)]

    time_kp=[]
    temp=np.array(np.concatenate((LH,RH,LF,RF), axis=None))
    if len(time_kp)==0:
        time_kp=temp
    else:
        time_kp=np.concatenate((time_kp,temp),axis=0)
    #array [9x1]-[num_fram, Lhx, Lhy, Rhx, Rhy, Lfx, Lfy, Rfx, Rfy]

    pose_store.append(time_kp)

    #call COM fucntion
    COM=get_COM(smoothed_pose,width,height)
    COM_store.append(COM)

    return time_kp, pose_mask, pose_store, COM_store

def get_COM(smoothed_pose,width,height):
    COM=np.zeros([1,2])
    keypoint_pos=[]
    mass_distri=np.array([8.2, 46.84, 3.25,3.25, 1.8,1.8, 0.65,0.65, 10.5,10.5, 4.75,4.75, 1.43,1.43]) #person model, Hall paper, adds to 99.8%
    mass_distri=np.expand_dims(mass_distri, axis=1) #vert array
    #keypoint pos
    keypoint_pos.append([int(smoothed_pose[0][0]*height), int(smoothed_pose[0][1]*width)]) #head
    temp_pos= [[int(smoothed_pose[11][0]*width), int(smoothed_pose[11][1]*height)],
                [int(smoothed_pose[12][0]*width), int(smoothed_pose[12][1]*height)],
                [int(smoothed_pose[23][0]*width), int(smoothed_pose[23][1]*height)],
                [int(smoothed_pose[24][0]*width), int(smoothed_pose[24][1]*height)]] #torso
    keypoint_pos.append(np.mean(temp_pos,axis=0))
    temp_pos= [[int(smoothed_pose[12][0]*width), int(smoothed_pose[12][1]*height)],
                [int(smoothed_pose[14][0]*width), int(smoothed_pose[14][1]*height)]] #L upper arm
    keypoint_pos.append(np.mean(temp_pos,axis=0))
    temp_pos= [[int(smoothed_pose[11][0]*width), int(smoothed_pose[11][1]*height)],
                [int(smoothed_pose[13][0]*width), int(smoothed_pose[13][1]*height)]] #R upper arm
    keypoint_pos.append(np.mean(temp_pos,axis=0))
    temp_pos= [[int(smoothed_pose[14][0]*width), int(smoothed_pose[14][1]*height)],
                [int(smoothed_pose[16][0]*width), int(smoothed_pose[16][1]*height)]] #L lower arm
    keypoint_pos.append(np.mean(temp_pos,axis=0))
    temp_pos= [[int(smoothed_pose[13][0]*height), int(smoothed_pose[13][1]*width)],
                [int(smoothed_pose[15][0]*height), int(smoothed_pose[15][1]*width)]] #R lower arm
    keypoint_pos.append(np.mean(temp_pos,axis=0))
    temp_pos= [[int(smoothed_pose[20][0]*height), int(smoothed_pose[20][1]*width)],
                [int(smoothed_pose[18][0]*height), int(smoothed_pose[18][1]*width)],
                [int(smoothed_pose[16][0]*height), int(smoothed_pose[16][1]*width)]] #L hand
    keypoint_pos.append(np.mean(temp_pos,axis=0))
    temp_pos= [[int(smoothed_pose[19][0]*height), int(smoothed_pose[19][1]*width)],
                [int(smoothed_pose[17][0]*height), int(smoothed_pose[17][1]*width)],
                [int(smoothed_pose[15][0]*height), int(smoothed_pose[15][1]*width)]] #R hand
    keypoint_pos.append(np.mean(temp_pos,axis=0))
    temp_pos= [[int(smoothed_pose[24][0]*height), int(smoothed_pose[24][1]*width)],
                [int(smoothed_pose[26][0]*height), int(smoothed_pose[26][1]*width)]] #L thigh
    keypoint_pos.append(np.mean(temp_pos,axis=0))
    temp_pos= [[int(smoothed_pose[23][0]*height), int(smoothed_pose[23][1]*width)],
                [int(smoothed_pose[25][0]*height), int(smoothed_pose[25][1]*width)]] #R lower arm
    keypoint_pos.append(np.mean(temp_pos,axis=0))
    temp_pos= [[int(smoothed_pose[26][0]*height), int(smoothed_pose[26][1]*width)],
                [int(smoothed_pose[28][0]*height), int(smoothed_pose[28][1]*width)]] #L calf
    keypoint_pos.append(np.mean(temp_pos,axis=0))
    temp_pos= [[int(smoothed_pose[25][0]*height), int(smoothed_pose[25][1]*width)],
                [int(smoothed_pose[27][0]*height), int(smoothed_pose[27][1]*width)]] #R calf
    keypoint_pos.append(np.mean(temp_pos,axis=0))
    temp_pos= [[int(smoothed_pose[28][0]*height), int(smoothed_pose[28][1]*width)],
                [int(smoothed_pose[32][0]*height), int(smoothed_pose[32][1]*width)],
                [int(smoothed_pose[30][0]*height), int(smoothed_pose[30][1]*width)]] #L foot
    keypoint_pos.append(np.mean(temp_pos,axis=0))
    temp_pos= [[int(smoothed_pose[27][0]*height), int(smoothed_pose[27][1]*width)],
                [int(smoothed_pose[29][0]*height), int(smoothed_pose[29][1]*width)],
                [int(smoothed_pose[31][0]*height), int(smoothed_pose[31][1]*width)]] #L hand
    keypoint_pos.append(np.mean(temp_pos,axis=0))
    #COM(1d)=(m1x1+m2x2)/(m1+m2)
    keypoint_COM=keypoint_pos*mass_distri #m1[x1,y1], element wise mult
    COM_x=np.sum(keypoint_COM[:,0])/np.sum(mass_distri)
    COM_y=np.sum(keypoint_COM[:,1])/np.sum(mass_distri)
    COM=np.array([COM_x,COM_y])
    COM=COM.astype(int)
    #print(COM)
    return COM #COM=2d[x,y] coords