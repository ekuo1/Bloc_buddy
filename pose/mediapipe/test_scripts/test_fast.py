# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

name = "blue_overhang"
frame_width = 1280
frame_height = 720

# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(f'../test_examples/videos/{name}.mp4').start()
time.sleep(1.0)

# start the FPS timer
fps = FPS().start()
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
writer = cv2.VideoWriter(f'output_{name}.mp4', fourcc, 30, (frame_width,frame_height) )   

# loop over frames from the video file stream
while fvs.more():
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale (while still retaining 3
	# channels)
	frame = fvs.read()

	if frame is None:
		break
	frame = cv2.resize(frame, (frame_width, frame_height))
	#frame = np.dstack([frame, frame, frame])
	
    # show the frame and update the FPS counter
	cv2.imshow("Frame", frame)
	writer.write(frame)
	cv2.waitKey(1)
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()
writer.release()