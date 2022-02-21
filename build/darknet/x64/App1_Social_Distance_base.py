#================================================================
#  To learn how to Develop Advance YOLOv4 Apps - Then check out:
#  https://augmentedstartups.info/yolov4release
#================================================================ 
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from itertools import combinations


def is_close(p1, p2):
    """
    #================================================================
    # 1. Purpose : Calculate Euclidean Distance between two points
    #================================================================    
    :param:
    p1, p2 = two points for calculating Euclidean Distance

    :return:
    dst = Euclidean Distance between two 2d points
    """
    dst = math.sqrt(p1**2 + p2**2)
    #=================================================================#
    return dst 


def convertBack(x, y, w, h): 
    #================================================================
    # 2.Purpose : Converts center coordinates to rectangle coordinates
    #================================================================  
    """
    :param:
    x, y = midpoint of bbox
    w, h = width, height of the bbox
    
    :return:
    xmin, ymin, xmax, ymax
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    """
    :param:
    detections = total detections in one frame
    img = image from detect_image method of darknet

    :return:
    img with bbox
    """
    #================================================================
    # 3.1 Purpose : Filter out Persons class from detections and get 
    #           bounding box centroid for each person detection.
    #================================================================
    if len(detections)>0:
        centroid_dict=dict()
        objectId=0
        for detections in detections:
            name_tag=str(detections[0].decode())#Solo check las personas taggeadas en coco file q tiene str en all the nombre
            if name_tag == 'person':
                x,y,w,h=detections[2][0], \
                        detections[2][1], \
                        detections[2][2], \
                        detections[2][3],
                xmin, ymin, xmax, ymax = convertBack(float(x),float(y),float(w),float(h))

    #=================================================================#

    #=================================================================
    # 3.2 Purpose : Determine which person bbox are close to each other
    #=================================================================            	
    red_zone_list = []
    red_line_list = []
    for (id1,p1), (id2,p2) in combinations(centroid_dict.items(),2):
        dx,dy=p1[0] - p2[1], p1[1] - p2[1]
        distance = is_close(dx,dy)
        if distance < 75.0:
            if id1 not in red_zone_list:
                red_zone_list.append(id1)
                red_line_list.append(p1[0:2])
            if id2 not in red_zone_list:
                red_zone_list.append(id2)
                red_line_list.append(p2[0:2])

    for idx, box in centroid_dict.items():
        if idx in red_line_list:
            cv2.rectangle(img,(box[2],box[3],box[4],box[5]), (255,0,0),2)
        else:
            cv2.rectangle(img,(box[2],box[3],box[4],box[5])(0,255,0),2)
	#=================================================================#

		#=================================================================
    	# 3.3 Purpose : Display Risk Analytics and Show Risk Indicators
    	#=================================================================        
        text = "People at Risk: %s" % str(len(red_zone_list)) 
        location=(10,25)
        cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2,cv2.LINE_AA)

        for check in range(0, len(red_zone_list)-1):
            start_point = red_line_list[check]
            end_point=red_line_list[check +1]
            check_line_x=abs(end_point[0]-start_point[0])
            check_line_y=abs(end_point[1]-start_point[1])
            if (check_line_x < 75) and (check_line_y <25):
                cv2.line(img, start_point, end_point, (255,0,0),2)
        #=================================================================#
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():
    """
    Perform Object detection
    """
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./Input/test5.mp4")   # <----- Replace with your video directory
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
    # print("Video Reolution: ",(width, height))

    out = cv2.VideoWriter(
            "./Demo/test5_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, # <----- Replace with your output directory
            (new_width, new_height))
    
    # print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(new_width, new_height, 3)
    
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        out.write(image)

    cap.release()
    out.release()
    print(":::Video Write Completed")

if __name__ == "__main__":
    YOLO()