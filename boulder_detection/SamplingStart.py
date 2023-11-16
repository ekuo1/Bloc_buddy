import cv2
from RouteDetector import mask_segmentation, routeDetector, showPredefinedroute, getFirstFrame, homographyMatrix
import numpy as np
import matplotlib.pyplot as plt

def is_good_homography(H, threshold=0.1):
    # checking that they are unit vectors
    if abs(H[2, 2] - 1) > threshold:
        return False
    
    # checking that they are unit vectors
    if abs(np.linalg.norm(H[0, :2]) - 1) > threshold or abs(np.linalg.norm(H[1, :2]) - 1) > threshold:
        return False
    
    # checking that the first two rows are orthogonal
    if abs(np.dot(H[0, :2], H[1, :2])) > threshold:
        return False
    
    return True

#input video location
name = 'pink_boulders'
video_path = f'C:/Users/Sarah/Documents/Uni/FYP/video/{name}.mp4'
output_path = 'C:/Users/Sarah/Documents/Uni/FYP/video/red_frames/'

#gets the first frame
first_frame = getFirstFrame(video_path)
imagemask = mask_segmentation(first_frame, 'red') #finding the red route
route_dict = routeDetector(first_frame, imagemask) #dictionary containing location of contours

##### for testing purposes I sampled at 3fps
cap = cv2.VideoCapture(video_path)

# Get video fps and frame size
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Define the frame interval to extract frames at 3 FPS
frame_interval = int(round(fps / 20))
count = 0
count2 = 0
# Start reading the video frame by frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Only save frames at 3 FPS
    if frame_count % frame_interval == 0:
        count2 += 1
        warped_frame, Hcv = homographyMatrix(first_frame, frame)
        if is_good_homography(Hcv) == False:
            count += 1
        warped_frame = cv2.cvtColor(warped_frame, cv2.IMREAD_GRAYSCALE)
        for contour in route_dict['red2']:
            cv2.drawContours(warped_frame, [contour],-1, (0,0,255),2, maxLevel=2)
        
        cv2.imwrite(output_path + f'frame_{frame_count:06d}.jpg', warped_frame)

    frame_count += 1

cap.release()

print("count1:", count)
print(count2)
