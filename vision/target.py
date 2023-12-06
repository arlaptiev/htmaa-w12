
# import the opencv library 
import cv2 
import time
from track_pose import track_pose
import numpy as np
  
# define a video capture object 
vid = cv2.VideoCapture(0) 

start_time = time.time()
frame_count = 0
fps_interval = 1.0  # Update FPS every 1 second
current_time = start_time
verbose = False
curr_index = 0

##PID
dt = 50
Kp_x = 1
Kd_x = 10
Kp_y = 1
Kd_y = 10

tolerance = 10

last_error_x = None
last_error_y = None

# text setup
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
font_thickness = 2
font_color = (0, 0, 255)  # BGR color format
text_position1 = (50, 100)
text_position2 = (50, 150)
text_position3 = (50, 200)
text_position4 = (50, 250)

def get_target_center(targ):
    ty = (targ['left_hip'][1] + targ['right_hip'][1] + targ['left_shoulder'][1] + targ['right_shoulder'][1])/4
    tx = (targ['left_hip'][0] + targ['right_hip'][0] + targ['left_shoulder'][0] + targ['right_shoulder'][0])/4
    return (tx, ty)

def run_targeting(center,targs, img = None):
    global curr_index
    draw_crosshair(img, center, color = (0, 0, 255))
    if curr_index >= len(targs):
        curr_index = 0
        return
    if target(center, get_target_center(targs[curr_index]), img=img):
        curr_index += 1
        if curr_index >= len(targs):
            curr_index = 0
        draw_crosshair(img, center, color = (0, 255, 0))
        

def target(center,targ, img=None):
    global last_error_x, last_error_y
    cx, cy = center
    error_x = cx - targ[0]
    error_y = cy - targ[1]

    if last_error_x is None and last_error_y is None:
        last_error_x = error_x
        last_error_y = error_y
    
    d_error_x = error_x - last_error_x
    d_error_y = error_y - last_error_y

    yaw = Kp_x * error_x + Kd_x * d_error_x
    pitch = Kp_y * error_y + Kd_y * d_error_y

    last_error_x = error_x
    last_error_y = error_y

    debug_output1 = f'Yaw: {yaw}, Pitch:{pitch}'
    debug_output2 = f'der_x: { Kd_x * d_error_x}, der_y: {Kd_y * d_error_y}'
    debug_output3 = f'P_x: {Kp_x * error_x}, P_y: {Kp_y * error_y}'
    debug_output4 = f'curr_index: {curr_index}'
    if img is not None:
        cv2.putText(img, debug_output1, text_position1, font, font_size, font_color, font_thickness)
        cv2.putText(img, debug_output2, text_position2, font, font_size, font_color, font_thickness)
        cv2.putText(img, debug_output3, text_position3, font, font_size, font_color, font_thickness)
        cv2.putText(img, debug_output4, text_position4, font, font_size, font_color, font_thickness)
    print(debug_output1)

    if abs(center[0] - targ[0]) < tolerance and abs(center[1] - targ[1]) < tolerance:
        return True
    return False

def draw_crosshair(img, center, color = (0, 0, 255)):
    cx, cy = center
    cx = int(cx)
    cy = int(cy)
    cross_size = 30  # Change the size of the cross as needed
    thickness = 5  # Thickness of the lines

    # Draw horizontal line
    cv2.line(img, (cx - cross_size, cy), (cx + cross_size, cy), color, thickness)

    # Draw vertical line
    cv2.line(img, (cx, cy - cross_size), (cx, cy + cross_size), color, thickness)





if __name__ == "__main__":
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
        frame, targs = track_pose(frame, multipose = True)
        center = (frame.shape[0]/2, frame.shape[1]/2)
        run_targeting(center, targs, img=frame)
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