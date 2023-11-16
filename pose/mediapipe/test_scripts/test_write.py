import cv2
name = "blue_overhang"
cap = cv2.VideoCapture(f'../test_examples/videos/{name}.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
print('fps :'+str(fps))
# Change the FPS whatever you want
FPS=30
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
writer = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width,frame_height) )   
   
if (cap.isOpened()== False):
  print("Error opening video stream or file")

while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    cv2.imshow('Frame',frame)
    writer.write( frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else:
    break
writer.release()
cap.release()
cv2.destroyAllWindows()