
# import the opencv library 
import cv2 
import time
from track_pose import track_pose
  
# define a video capture object 
vid = cv2.VideoCapture(0) 

start_time = time.time()
frame_count = 0
fps_interval = 1.0  # Update FPS every 1 second
current_time = start_time

verbose = False
while(True): 
    frame_start_time = time.time()
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    if verbose: print("vid.read() :", time.time() - frame_start_time)

    frame_start_time = time.time()
    frame = cv2.flip(frame, 1)
    if verbose: print("cv2.flip :", time.time() - frame_start_time)

    frame_start_time = time.time()
    frame,_ = track_pose(frame, multipose = True)
    if verbose: print("track_pose:", time.time() - frame_start_time)

    # Display the resulting frame 
    frame_start_time = time.time()
    cv2.imshow('frame', frame) 
    if verbose: print("imshow:", time.time() - frame_start_time)
    frame_count += 1
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    frame_start_time = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    if verbose: print("waitKey:", time.time() - frame_start_time)

    elapsed_time = time.time() - current_time
    if elapsed_time > fps_interval:
        fps = frame_count / elapsed_time
        if verbose: print(f"FPS: {fps:.2f}")
        current_time = time.time()
        frame_count = 0
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 