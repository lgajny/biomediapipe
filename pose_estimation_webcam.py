# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 14:04:41 2023

@author: gajny
For now, this script is a joint use of https://github.com/Pawandeep-prog/mediapipe
this script will in the end be a real-time kinematic analysis script
"""

import mediapipe as mp
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True,min_detection_confidence = 0.6, min_tracking_confidence=0.6);

facmesh = mp.solutions.face_mesh
face = facmesh.FaceMesh(static_image_mode=True, min_tracking_confidence=0.6, min_detection_confidence=0.6)

hands = mp.solutions.hands
hands_mesh = hands.Hands(static_image_mode=True, min_detection_confidence=0.6)

draw = mp.solutions.drawing_utils

while True:

	_, frm = cap.read()
	rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

	op_pose = pose.process(rgb)
	op_face = face.process(rgb)
	op_hands = hands_mesh.process(rgb)
        
	draw.draw_landmarks(frm,op_pose.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 255), circle_radius=1))
	
	if op_face.multi_face_landmarks:
		for i in op_face.multi_face_landmarks:
			print(i.landmark[0].y*480)
			draw.draw_landmarks(frm, i, facmesh.FACEMESH_CONTOURS, landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 255), circle_radius=1))

	if op_hands.multi_hand_landmarks:
		for i in op_hands.multi_hand_landmarks:
			draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS, 
				landmark_drawing_spec=draw.DrawingSpec(color = (255, 0,0),circle_radius=2, thickness=1),
				connection_drawing_spec=draw.DrawingSpec(thickness=3, color=(0,0,255)))
            
	cv2.imshow("window", frm)

	if cv2.waitKey(1) == 27:
		cap.release()
		cv2.destroyAllWindows()
		break
    