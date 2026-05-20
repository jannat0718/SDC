import sys
import os
import cv2
import can
import threading
import argparse
import logging
import time
import math
import numpy as np
import csv
import tqdm
import video
import itertools
import matplotlib.pyplot as plt
import configparser 
import struct
from datetime import datetime
from datetime import timedelta
from collections import namedtuple
from typing import Optional, Dict, List, Literal, Tuple
from queue import Queue
from time import sleep
import json
from lane_detection_class import Point




CAN_MESSAGE_SENDING_SPEED = 0.02


class CanListener:
    """
    A can listener that listens for specific messages and stores their latest values.
    """

    _id_conversion = {
        0x110: 'brake',
        0x220: 'steering',
        0x330: 'throttle',
        0x1e5: 'steering_sensor'
    }

    def __init__(self, bus: can.Bus):
        self.bus = bus
        self.running = False
        self.thread = threading.Thread(target = self._listen, args = (), daemon = True)
        self.data : Dict[str, List[int]] = {name: None for name in self._id_conversion.values()}
    
    def start_listening(self):
        self.running = True
        self.thread.start()
    
    def stop_listening(self):
        self.running = False
    
    def get_new_values(self):
        values = self.data
        return values

    def _listen(self):
        while self.running:
            message: Optional[can.Message] = self.bus.recv(.5)
            if message and message.arbitration_id in self._id_conversion:
                self.data[self._id_conversion[message.arbitration_id]] = message.data
                
                     
                
class CanWorker:
    """
    A worker that writes can-message values to disk.
    """

    def __init__(self, can_queue: Queue, folder: str):
        self.queue = can_queue
        self.thread = threading.Thread(target = self._process, args = (), daemon = True)
        self.folder_name = folder
        self.file_pointer = open(os.path.join(self.folder_name, f'recording.csv'), 'w')
        print('Timestamp|Steering|SteeringSpeed|Throttle|Brake|SteeringSensor', file = self.file_pointer)
    
    def start(self):
        self.thread.start()
    
    def stop(self):
        self.queue.join()
        self.file_pointer.close()
    
    def put(self, data):
        #print(data)
        self.queue.put(data)
    
    def _process(self):
        while True:
            timestamp, values = self.queue.get()
            steering = str(struct.unpack("f", bytearray(values["steering"][:4]))[0]) if values["steering"] else " "
            steering_speed = str(struct.unpack(">I", bytearray(values["steering"][4:]))[0]) if values["steering"] else " "
            throttle = str(values["throttle"][0]/100) if values["throttle"] else " "
            brake = str(values["brake"][0]/100) if values["brake"] else " "
            if values["steering_sensor"]:
                steering_sensor = (values["steering_sensor"][1] << 8 | values["steering_sensor"][2])
                steering_sensor -= 65536 if steering_sensor > 32767 else 0
            else:
                steering_sensor = " "
            print(f'{timestamp}|{steering}|{steering_speed}|{throttle}|{brake}|{steering_sensor}', file=self.file_pointer)
            self.queue.task_done()
            
            
class ImageWorker:
    """
    A worker that writes images to disk.
    """

    def __init__(self, image_queue: Queue, folder: str):
        self.queue = image_queue
        self.thread = threading.Thread(target = self._process, args = (), daemon = True)
        self.folder: str = folder
    
    def start(self):
        self.thread.start()
    
    def stop(self):
        self.queue.join()

    def put(self, data):
        self.queue.put(data)
        
    def _process(self):
        while True:
            filename, image_type, image = self.queue.get()
            cv2.imwrite(os.path.join(self.folder, image_type, f'{filename}.png'), image)
            self.queue.task_done()
            
            

class CanMessage:
    """
    A messenger that sends messages to the actuators based on the input.
    """
    
    def __init__(self, bus: can.Bus, can_message_queue: Queue):
        self.queue = can_message_queue
        self.bus = bus
        self.reset_states()
        self.thread = threading.Thread(target = self.control_kart, args = (), daemon = True)
        self.steer = 0
        self.throttle = 0
        self.brake = 0
        self.kart_running = False

    
    def reset_states(self):
        self.brake_message = can.Message(arbitration_id=0x110, data=[0, 0, 0, 0, 0, 0, 0, 0], is_extended_id=False)
        #self.brake_task = self.bus.send(self.brake_message)
        self.brake_task = self.bus.send_periodic(self.brake_message, CAN_MESSAGE_SENDING_SPEED)
        
        self.steer_message = can.Message(arbitration_id=0x220, data=[0, 0, 0, 0, 0, 0, 0, 0], is_extended_id=False)
        #self.steer_task = self.bus.send(self.steer_message)
        self.steer_task = self.bus.send_periodic(self.steer_message, CAN_MESSAGE_SENDING_SPEED)
        
        self.motor_message = can.Message(arbitration_id=0x330, data=[0, 0, 0, 0, 0, 0, 0, 0], is_extended_id=False)
        #self.motor_task = self.bus.send(self.motor_message)
        self.motor_task = self.bus.send_periodic(self.motor_message, CAN_MESSAGE_SENDING_SPEED)
        
    def start(self):
        self.kart_running = True
        self.thread.start()
    
    def put(self, data):
        self.queue.put(data)
    
    
    def stop(self):
        self.queue.join()
        self.kart_running = False
        
              
    def control_kart(self):
        while self.kart_running:
            throttle, steer, brake = self.queue.get()
            self.steer = steer
            self.throttle = throttle
            self.brake = brake

            steer = np.fmax(np.fmin(self.steer, 1.0), -1.0)
            throttle = np.fmax(np.fmin(self.throttle, 1.0), 0)
            brake = np.fmax(np.fmin(self.brake, 1.0), 0)

            #print(steer, throttle, brake)
            
            self.steer_value = float(steer)
            self.motor_value = int(throttle * 100)
            self.brake_value = int(brake * 100)

            self.steer_message.data = list(bytearray(struct.pack("f", self.steer_value))) + [0]*4
            #self.bus.send(self.steer_message)
            self.steer_task.modify_data(self.steer_message)


            self.brake_message.data = [self.brake_value] + [0]*7
            #self.bus.send(self.brake_message)
            self.brake_task.modify_data(self.brake_message)


            self.motor_message.data = [self.motor_value] + [0] + [1] + [0]*5
            #self.bus.send(self.motor_message)
            self.motor_task.modify_data(self.motor_message)

            #print('msg')
            self.queue.task_done()
        
            
def main():
    
    print('Initializing...', file=sys.stderr)
    camera_on = True
    mean_avg = True
    
    if camera_on:
        print('Fetching images from camera')
        cameras = initialize_cameras()

    print('Creating folders...', file=sys.stderr)
    
    path = "save_path.json"
    with open(path, 'r') as fpath:
        dir_path = json.load(fpath)

    save_path = list(dir_path.values())[0] #Need to change indexing to '1' if needed to store in USB drive
    folder_path = ['Image_recordings/', 'Video_recordings/', 'Lidar_recordings/', 'Image_Lidar_recordings/']

    for folder in folder_path:
        if not os.path.exists(save_path + folder):
            os.mkdir(save_path + folder)

    recording_folder = save_path + folder_path[0] + "recording " + datetime.now().strftime("%d-%m-%Y %H-%M-%S") #Choose right index
    if not os.path.exists(recording_folder):
        os.mkdir(recording_folder)
        if camera_on:
            for subdir in cameras.keys():
                os.mkdir(os.path.join(recording_folder, subdir))
    
    
    bus = initialize_can()
    can_listener = CanListener(bus)
    can_listener.start_listening()
    # image_queue = Queue()
    # image_worker = ImageWorker(image_queue, recording_folder)
    # image_worker.start()
    can_worker = CanWorker(Queue(), recording_folder)
    can_worker.start()  
    
    can_message = CanMessage(bus, Queue())
    can_message.start()
    
    detect = Point()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color_r = (0, 0, 255)
    color_b = (255, 0, 0)
    thickness = 2
    
    throttle_straight = 1
    throttle_curve = 0.7
    brake = 0
    print('High speed setting')

    print('Recording...', file=sys.stderr)
    frames: Dict[str, cv2.Mat] = dict() 
    
    print('\n############## Initializing the motors ##############', file=sys.stderr)

    if not camera_on:
        image_dir = '/home/sdc/SDC_UT/Programs/Image_recordings/recording 02-05-2023 14-33-43/front/'
        image_dir_rt = '/home/sdc/SDC_UT/Programs/Image_recordings/recording 02-05-2023 14-33-43/right/'
        image_dir_lt = '/home/sdc/SDC_UT/Programs/Image_recordings/recording 02-05-2023 14-33-43/left/'
        
        img_lst = []
        
        j=0

        for fname in tqdm.tqdm(itertools.islice(sorted(os.listdir(image_dir)),None, None, 1)):
            img_lst.append(fname)
        len_img = len(img_lst)
        print(len_img)
    
    
    start_time = time.time()
    n = 7.5   #no. of future minutes for change
    time_check = 0
    current_time = datetime.now()
    future_time = current_time + timedelta(minutes=n)
    print('Current time is:{}'.format(current_time))
    print('Future time for change is:{}'.format(future_time))
      
    
    
    try:
        while True:
            
            if camera_on:
                ok_count = 0
                values = can_listener.get_new_values()
                timestamp = time.time()
                for side, camera in cameras.items():
                    ok, frames[side] = camera.retrieve()
                    ok_count += ok
                if ok_count == len(cameras):
                    # for side, frame in frames.items():
                    #     image_worker.put((timestamp, side, frame))
                    can_worker.put((timestamp, values))
                    front_frame = frames["front"]
                    right_frame = frames["right"]
                    left_frame = frames["left"]
                for camera in cameras.values():
                    camera.grab()
                timestamp_name = str(timestamp) + '.png'
        
            else:
                if j < len(img_lst):
                    timestamp = time.time()
                    values = can_listener.get_new_values()
                    can_worker.put((timestamp, values))
                    front_frame = cv2.imread(os.path.join(image_dir + img_lst[j]))
                    right_frame = cv2.imread(os.path.join(image_dir_rt + img_lst[j]))
                    left_frame = cv2.imread(os.path.join(image_dir_lt + img_lst[j]))
                    timestamp_name = img_lst[j]
                    j = j+1
                else:
                    break
            
            
            right_steer = detect.right_steer(right_frame)
            left_steer = detect.left_steer(left_frame)            
            
                        
            current_time = datetime.now()
            if current_time > future_time and time_check == 0:
                print(current_time)
                print('Time elapsed, changing to lower speed')
                throttle_straight = 0.8
                throttle_curve = 0.5
                time_check = 1
            
            
            if (right_steer != 0 and left_steer != 0):   # Lines detected in both right & left cameras - applicable only to the last segment of the SDC track
                steer_limited = right_steer
                current_loop_time = datetime.now()
                if current_loop_time > future_time:
                    right_steer = 0
                    left_steer = 0
            
            
            #execute the front camera part only if right & left steer are not zero
            if (right_steer == 0 and left_steer == 0) : 
            
                # Get lines after applying blurr and hough transform
                lines = detect.line_detect(front_frame)

                if lines is None:
                    print('No hough lines detected')
                    continue

                # Get slope and intercept for each detected line
                lines_arr = lines[:,0,:]
                slopes, intercepts = detect.get_slope_intercept(lines_arr)

                # Threshold based on line point and slope
                lines, intercepts, slopes = detect.line_point_thres(slopes, intercepts, lines_arr)

                # Threshold based on lines slope
                fil_lines, fil_intercepts, fil_slopes = detect.slope_thres(lines, intercepts, slopes, 0.05)

                if fil_lines.shape[0] == 0 :
                    fil_lines, fil_intercepts, fil_slopes = detect.slope_thres(lines, intercepts, slopes, 0.03)

                if fil_lines.shape[0] == 0 :
                    print('No lines detected after slope thresholding')
                    continue

                # Threshold based on lines intercept
                final_lines, final_intercepts, final_slopes = detect.inter_thes(fil_lines, fil_intercepts, fil_slopes)

                if final_lines.shape[0] == 0:
                    print('No lines detected after intercept thresholding')
                    final_lines = fil_lines
                    final_intercepts = fil_intercepts
                    final_slopes = fil_slopes

                # Group filtered lines based on their slope
                total_lines = detect.line_sort(final_slopes, final_lines, final_intercepts)

                # Merge similar lines together
                if mean_avg:
                    lines_coord = []
                    for i in range(len(total_lines)):
                        t_lines = total_lines[i]
                        f_x1 = np.mean(np.array(t_lines)[:,0], dtype=int)
                        f_y1 = np.mean(np.array(t_lines)[:,1], dtype=int)
                        f_x2 = np.mean(np.array(t_lines)[:,2], dtype=int)
                        f_y2 = np.mean(np.array(t_lines)[:,3], dtype=int)
                        lines_coord.append([f_x1, f_y1, f_x2, f_y2])

                else:
                    lines_coord = []
                    for i in range(len(total_lines)):
                        t_lines = total_lines[i]
                        f_x1 = np.min(np.array(t_lines)[:,0])
                        f_y1 = np.max(np.array(t_lines)[:,1])
                        f_x2 = np.max(np.array(t_lines)[:,2])
                        f_y2 = np.min(np.array(t_lines)[:,3])
                        lines_coord.append([f_x1, f_y1, f_x2, f_y2])

                lines_coord = np.array(lines_coord)

                # Draw final lines
                for i in range(lines_coord.shape[0]):
                    cv2.line(front_frame, lines_coord[i][:2], lines_coord[i][2:], (0, 0, 255), 2)

                width = front_frame.shape[1]
                height = front_frame.shape[0]

                xl_1,yl_1,xl_2,yl_2 = lines_coord[0]
                xr_1,yr_1,xr_2,yr_2 = lines_coord[1]

                A_x, A_y = xl_1, yl_1 #left bottom point
                B_x, B_y = xl_2, yl_2 #left top point
                C_x, C_y = xr_1, yr_1 #right top point
                D_x, D_y = xr_2, yr_2 #right bottom point

                POI_x, POI_y = detect.lineLineIntersection(A_x, A_y, B_x, B_y, C_x, C_y, D_x, D_y)   #Line AB and line CD intersection point

                x2, y2 = POI_x, POI_y
                x1, y1 = (width/2, height)  #check

                angle_centre_line = np.deg2rad(90)

                steer_value_rad = np.arctan2((y1-y2),(x1-x2))

                steer_rad = steer_value_rad - angle_centre_line

                steer_deg = np.rad2deg(steer_rad)

                steer_limited = steer_rad * (180.0 / 60.0 / np.pi) #considering maximum steer angle as 60deg and it's corresponding radians and so dividing the input radian steer by this

            
            
            elif left_steer ==0 and right_steer != 0:
                steer_limited = right_steer
                
            elif right_steer ==0 and left_steer != 0:
                steer_limited = left_steer
            
            # elif (right_steer != 0 and left_steer != 0):   ##both right&left cameras lines detection
            #     steer_limited = right_steer

            
            if steer_limited >= 0.6 or steer_limited <= -0.6:
                throttle = throttle_curve
            else:
                throttle = throttle_straight
            

            can_message.put((throttle, steer_limited, brake))
            
            
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
        pass
    
    #print('Stopping the recording...', file=sys.stderr)
    
    end_time = time.time()
    
    time_diff = end_time - start_time 
    print(time_diff)
    
    can_listener.stop_listening()
    
    if camera_on:
        for camera in cameras.values():
            camera.release()
    
    # image_worker.stop()
    can_worker.stop()                         
                                 
    can_message.stop()
    can_message.reset_states()
    
    sleep(2)
    print('All motors are reset...')
    print('\n#######################################################\n', file=sys.stderr)



def initialize_can() -> can.Bus:
    """
    Set up the can bus interface and apply filters for the messages we're interested in.
    """
    bus = can.Bus(interface='socketcan', channel='can0', bitrate=500000)
    bus.set_filters([
        {'can_id': 0x110, 'can_mask': 0xfff, 'extended': False}, # Steering
        {'can_id': 0x220, 'can_mask': 0xfff, 'extended': False}, # Throttle
        {'can_id': 0x330, 'can_mask': 0xfff, 'extended': False}, # Brake
        {'can_id': 0x1e5, 'can_mask': 0xfff, 'extended': False}, # Steering sensor
    ])
    return bus

def initialize_cameras() -> Dict[str, cv2.VideoCapture]:
    """
    Initialize the opencv camera capture devices.
    """
    config: video.CamConfig = video.get_camera_config()
    if not config:
        print('No valid video configuration found!', file=sys.stderr)
        exit(1)
    cameras: Dict[str, cv2.VideoCapture] = dict()
    for camera_type, path in config.items():
        capture = cv2.VideoCapture(path)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        capture.set(cv2.CAP_PROP_FOCUS, 0)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #important to set right codec to enable 60fps
        capture.set(cv2.CAP_PROP_FPS, 30) #make 60 to enable 60FPS
        cameras[camera_type] = capture
    return cameras

if __name__ == '__main__':
    main()