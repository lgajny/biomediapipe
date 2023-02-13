# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:12:31 2023
Offline use of https://github.com/Pawandeep-prog/mediapipe
@author: gajny
"""

import mediapipe as mp
import numpy as np
import cv2

cap = cv2.VideoCapture('temp_video_1671053153962.mp4')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True,min_detection_confidence = 0.6, min_tracking_confidence=0.6);

facmesh = mp.solutions.face_mesh
face = facmesh.FaceMesh(static_image_mode=True, min_tracking_confidence=0.6, min_detection_confidence=0.6)

hands = mp.solutions.hands
hands_mesh = hands.Hands(static_image_mode=True, min_detection_confidence=0.6)

draw = mp.solutions.drawing_utils
out = cv2.VideoWriter('//intram/paris/Labo/IBHGC/ECHANGE/Pour_Laurent/snowboard/demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (1079,1120))


for i in range(100):
    _, frm = cap.read()
    if frm is not None:
        frm = frm[800:1920,1:1080]
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        op_pose = pose.process(rgb)
        # op_face = face.process(rgb)
        # op_hands = hands_mesh.process(rgb)
        draw.draw_landmarks(frm,op_pose.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 255), circle_radius=1))
        	
        # if op_face.multi_face_landmarks:
        # 	for i in op_face.multi_face_landmarks:
        # 		print(i.landmark[0].y*480)
        # 		draw.draw_landmarks(frm, i, facmesh.FACEMESH_CONTOURS, landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 255), circle_radius=1))
        
        # if op_hands.multi_hand_landmarks:
        # 	for i in op_hands.multi_hand_landmarks:
        # 		draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS, 
        # 			landmark_drawing_spec=draw.DrawingSpec(color = (255, 0,0),circle_radius=2, thickness=1),
        # 			connection_drawing_spec=draw.DrawingSpec(thickness=3, color=(0,0,255)))
                
        cv2.imshow("window", frm)
        out.write(frm)
    
    if cv2.waitKey(1) == 27:
    	cap.release()
    	cv2.destroyAllWindows()
    	break
        
out.release()