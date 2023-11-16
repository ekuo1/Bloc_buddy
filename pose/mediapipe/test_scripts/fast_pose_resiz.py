# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

name = "red_crimps"

# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(f'../test_examples/videos/{name}.mp4').start()
time.sleep(1.0)

frame_width = int(fvs.stream.get(3))
frame_height = int(fvs.stream.get(4))
width = int(frame_width * (640 / frame_height))
height = 640

# start the FPS timer
fps = FPS().start()
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
writer = cv2.VideoWriter(f'output_{name}.mp4', fourcc, 10, (width,height) )   
count = 0
with mp_pose.Pose(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5, model_complexity = 1) as pose:

	# loop over frames from the video file stream
	while fvs.more():
		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale (while still retaining 3
		# channels)
		frame = fvs.read()
		if frame is None:
			break
		
		count += 1 
		if count == 3:
			frame.flags.writeable = False
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			results = pose.process(frame)	

			frame.flags.writeable = True
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

			if results.pose_landmarks:
			# Display the found normalized landmarks.
				#print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}') 

				mp_drawing.draw_landmarks(
					frame,
					results.pose_landmarks,
					mp_pose.POSE_CONNECTIONS,
					landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
			
			# show the frame and update the FPS counter
			frame = cv2.resize(frame, (width, height))
			#frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
			writer.write(frame)

			# resizing the frame seemed to slow it down?
			cv2.imshow("Frame", frame)
			cv2.waitKey(1)

			count = 0
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