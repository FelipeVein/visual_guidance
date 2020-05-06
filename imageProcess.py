#!/usr/bin/env python
# -*- coding: utf-8 -*-

#############################################
#                                           #
# Image process class for Agricultural   	#
# robots                    				#
#                                           #
# Author: Adalberto Oliveira                #
# Mastering in robotic - PUC-Rio            #
# Version: 1.0                              #
# Date: 4-22-2019                           #
#                                           #
#############################################

import rospy, time, angles, math, tf2_geometry_msgs, tf, sys, cv2
import numpy as np
from geometry_msgs.msg import Twist, Pose2D, PointStamped 
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import tf2_ros as tf2


def hello_world(msg):

    print msg

#--------------------------------------------------------------------

def cartesian_ctrl(position, destination, K):

    # remaping input variables
    x = destination.point.x
    y = destination.point.y
    theta = position.theta
    
    #v = -K[0] * (x * math.cos(theta) + y * math.sin(theta))
    v = K[0] * (math.sqrt(x **2 + y**2))
    alpha = math.atan2(y,x)
    w = K[1] * alpha 
    U_uni = [v, w]
    
    return U_uni

#--------------------------------------------------------------------

def polar_ctrl(position, destination, K_polar):
    # Polar coordinate control function

    # remaping input variables
    x = destination.point.x
    y = destination.point.y 
    theta = position.theta
    theta_goal = round(destination.point.z, 3)

    # Remaping the gain variables
    k_rho = K_polar[0]
    k_alpha = K_polar[1]
    k_beta = K_polar[2]

    # Computing the coordinate transformation
    rho = math.sqrt(x **2 + y**2)
    alpha = math.atan2(y, x)
    beta = round(angles.shortest_angular_distance(theta_goal,theta),3) 

    # Defining the position threshold 
    if abs(rho) < 0.15:
        alpha = 0
        rho = 0

    # Computin the control law
    v = (k_rho * rho)
    w = ((k_alpha * alpha) + (k_beta * beta))

    agl = [rho, alpha, beta]    # Secondarie monitiring parameters
    U_uni = [v,w, agl]

    return U_uni

#--------------------------------------------------------------------

def polar_ctrl_avoid(position, ala_position, destination, K_polar):
    # Polar coordinate control function with obstacle avoidance

    # Local constants
    radius = 0.3
    threshold = 1

    # remaping input variables
    x = destination.point.x
    y = destination.point.y 
    theta = position.theta
    theta_goal = round(destination.point.z, 3)
    x_ala = ala_position.point.x
    y_ala = ala_position.point.y 

    # Remaping the gain variables
    k_rho = K_polar[0]
    k_alpha = K_polar[1]
    k_beta = K_polar[2]
    k_ala_v = K_polar[3]
    k_ala_w = K_polar[4]


    # Computing the coordinate transformation
    rho = math.sqrt(x **2 + y**2)
    alpha = math.atan2(y,x)
    beta = round(angles.shortest_angular_distance(theta,theta_goal),2)
    
    # Computing the avoidance parameter
    ala = math.sqrt(x_ala **2 + y_ala**2) - radius
    alpha_ala = (math.atan2(y_ala,x_ala)) 
    
    if alpha_ala > 0: 
        alpha_ala-= (math.pi/2)
    elif alpha_ala < 0:
        alpha_ala+= (math.pi /2)
    else:
        pass
    
    avoid = 1 / ((ala )**3 - (ala**2 * threshold))
    if avoid < 0: avoid = 0

    # Defining the position threshold 
    if abs(rho) < 0.15:
        alpha = 0
        rho = 0

    # Computing the control law
    v = (k_rho * rho)
    w = (k_alpha * alpha) + (k_beta * beta)

    # Avoidance action theshold
    if ala <= threshold:
        v = abs((k_rho * rho) - (ala * k_ala_v))
        w = (k_alpha * alpha) + (k_beta * beta) + (k_ala_w * (alpha_ala))


    agl = [rho, alpha, beta, avoid, ala, alpha_ala]
    U_uni = [v,w, agl]

    return U_uni

#--------------------------------------------------------------------

def polar_ctrl_hybrid(position, destination, img_theta, K_polar):
    # Polar coordinate control function

    # remaping input variables
    x = destination.point.x
    y = destination.point.y 
    theta = position.theta
    theta_goal = round(destination.point.z, 3)

    # Remaping the gain variables
    k_rho = K_polar[0]
    k_alpha = K_polar[1]
    k_beta = K_polar[2]
    k_img = K_polar[3]

    # Computing the coordinate transformation
    rho = math.sqrt(x **2 + y**2)
    alpha = math.atan2(y, x)
    beta = round(angles.shortest_angular_distance(theta_goal,theta),3) 
    

    # Defining the position threshold 
    if abs(rho) < 0.15:
        alpha = 0
        rho = 0
        k_img = 0


    # Computin the control law
    v = k_rho * rho
    w = (k_alpha * alpha) + (k_beta * beta) + (k_img * img_theta)

    agl = [rho, alpha, beta, img_theta]    # Secondarie monitiring parameters
    U_uni = [v,w, agl]

    return U_uni

#--------------------------------------------------------------------

def polar_ctrl2(position, destination, K_polar):
    # Polar coordinate control function 

    # remaping input variables
    delta_x = destination.point.x
    delta_y = destination.point.y
    theta = position.theta
    delta_theta = round(destination.point.z, 3)
    
    k_rho = K_polar[0]
    k_alpha = K_polar[1]
    k_beta = K_polar[2]

    rho = math.sqrt(math.pow(delta_x,2) + math.pow(delta_y,2))
    alpha =  math.atan2(delta_y, delta_x) + math.pi
    beta = alpha + theta
    beta = angles.shortest_angular_distance(alpha,theta) 
    
    print 'rho: ', rho
    print 'alpha: ', alpha
    print 'beta: ', beta,'----\n'

    v = k_rho * rho * math.cos(alpha)
    w = (k_alpha * alpha) + (k_rho * ((math.sin(alpha)*math.cos(alpha))/alpha) *(alpha + (k_beta * beta)))

    U_uni = [v,w]

    return U_uni

#--------------------------------------------------------------------

def PID(K_pid, PID):

    last_proportional = PID[0]
    integral = PID[1]
    
    derivative = PID[2]
    integral = PID[1] + PID[3]
    derivative = PID[3] - PID[0]
    last_proportional = PID[3]
    if integral > 300: integral = 0
    #integral = integral//2

    pid =  (K_pid[0] * PID[3]) + (K_pid[1] * integral) + (K_pid[2] * derivative)

    PID = [last_proportional, integral, derivative, pid]
    
    return PID

#--------------------------------------------------------------------

def get_img(cv_image):
    # Image process method for color space transformation (BGR --> OTHA)
    
    rows, col, channels = cv_image.shape

    # Spliting image layers
    R = cv_image[0:rows, 0:col,2]
    G = cv_image[0:rows, 0:col,1]
    B = cv_image[0:rows, 0:col,0]
    
    # OTA color space convertion
    I1_prime = R-G
    I2_prime = R-B
    I3_prime = (2*G - R - B)/2
    
    return np.copy(cv_image), I1_prime, I2_prime, I3_prime

#--------------------------------------------------------------------

def get_centroid(cv_output, I1_prime, I2_prime, I3_prime):
    # Image process method for feature extration

    bk_h,bk_w = cv_output.shape

    # Image blurring 
    #blur1 = cv2.GaussianBlur(I1_prime,(21,21),20)
    #blur2 = cv2.GaussianBlur(I3_prime,(21,21),20)

    # Binary mask genteration
    #th,mask1 = cv2.threshold(blur1,
    #                        0,255,
    #                        cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    
    #th,mask2 = cv2.threshold(blur2,
    #                        0,255,
    #                        cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #mask2 = ~mask2
    #mask_out = ~(mask1 & mask2)
    #mask_out = ~mask1

    mask_out = cv_output

    # Fiding mask contours
    contour = cv2.findContours(mask_out, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contour = contour[1]
    
    # Contours parameters
    area_1 = 0
    area_2 = 0
    moment_1 = []
    moment_2 = []
    Cont_1 = []
    Cont_2 = []
    centroid_1 = [0,0]
    centroid_2 = [0,0]   

    # Fiding great area in the mask
    for c in contour:
        M = cv2.moments(c)
        if (M["m00"] > area_2):
            if (M["m00"] > area_1):
                area_2 = area_1
                moment_2 = moment_1
                moment_1 = M
                
                area_1 = M["m00"]
                Cont_2 = Cont_1
                Cont_1 = [c]
            else:
                area_2 = M["m00"]
                moment_2 = M
                Cont_2 = [c]
    #print area_1
    if area_1 > 1000: # 1000:
        centroid_1[0] = int(moment_1["m10"]/moment_1["m00"])
        centroid_1[1] = int(moment_1["m01"]/moment_1["m00"])
        cv2.circle(cv_output, (centroid_1[0], centroid_1[1]), 7, (255,0,0),-1)
        cv2.drawContours(cv_output, Cont_1 ,-1,(0,255,0),2)
    else:
        centroid_1[0] = 0
        centroid_1[1] = 0
    
    # Find higher point in the contour
    K = Cont_1[0][0]
    s = max(K, key=lambda item: (-item[0], item[1]))
    cv2.circle(cv_output, (s[0], s[1]), 7, (255,0,0),-1)
    cv2.line(cv_output, (320,0),(320,480), (255,0,0),2)

    #print(Cont_1)



    mask_show = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)
    mask_show2 = cv2.fillPoly(np.zeros_like(mask_show), pts = Cont_1, color=(255,255,255))
    
    return cv_output, mask_show, centroid_1, mask_show2

#--------------------------------------------------------------------
    
def process_image_2():
    global cmd_vel
    
    # Image resize parameters
    W = int(640*0.7)
    H = int(480*0.7)


    # Image acquisition
    cv_output, I1_prime, I2_prime, I3_prime = get_img(0)
    #rows, col, channels = cv_raw.shape
    #cv_output = cv_raw.copy()
    roi = cv_output[200:430,:,:]
    bk_h,bk_w,bk_d = cv_output.shape
    black_img = np.zeros((bk_h,bk_w))
    

    # Image blurring 
    blur1 = cv2.GaussianBlur(I1_prime,(21,21),20)
    blur2 = cv2.GaussianBlur(I3_prime,(21,21),20)

    # Binary mask genteration
    th,mask1 = cv2.threshold(blur1,
                            0,255,
                            cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    
    th,mask2 = cv2.threshold(blur2,
                            0,255,
                            cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #mask2 = ~mask2
    #mask_out = ~(mask1 & mask2)
    mask_out = ~mask1
    mask_out = mask_out[200:430,:]

    # Fiding mask contours
    contour = cv2.findContours(mask_out, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contour = contour[1]
    
    # Contours parameters
    area_1 = 0
    area_2 = 0
    moment_1 = []
    moment_2 = []
    Cont_1 = []
    Cont_2 = []
    centroid_1 = [0,0]
    centroid_2 = [0,0]   

    # Fiding great area in the mask
    for c in contour:
        M = cv2.moments(c)
        if (M["m00"] > area_2):
            if (M["m00"] > area_1):
                area_2 = area_1
                moment_2 = moment_1
                moment_1 = M
                
                area_1 = M["m00"]
                Cont_2 = Cont_1
                Cont_1 = [c]
            else:
                area_2 = M["m00"]
                moment_2 = M
                Cont_2 = [c]

    
    if area_1 > 1000:
        centroid_1[0] = int(moment_1["m10"]/moment_1["m00"])
        centroid_1[1] = int(moment_1["m01"]/moment_1["m00"])
        cv2.circle(roi, (centroid_1[0], centroid_1[1]), 7, (255,0,0),-1)
        cv2.drawContours(roi, Cont_1 ,-1,(0,255,0),2)
        cv2.drawContours(black_img, Cont_1 ,-1,(0,255,0),2)
    K = Cont_1[0][0]

    s = max(K, key=lambda item: (-item[0], item[1]))
    #print 'Area 1: ', area_1, ' Area 2: ', area_2, ' Max Contour elemts: ',s
    
    cv2.circle(cv_output, (s[0], s[1]), 7, (255,0,0),-1)
    cv2.line(cv_output, (320,0),(320,480), (255,0,0),2)

    cv2.fillPoly(black_img, Cont_1, 255)
    cv_output[200:430,:,:] = roi
    fonte = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(cv_output, 
                'Robot view' ,(100, 450), 
                fonte, 1.5, (0,0,255),2)    
    
    mask_show = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)
    
    cv2.putText(mask_show, 
                'Mask view' ,(100, 450), 
                fonte, 1.5, (0,255,0),2)    
    '''
    cv2.putText(cv_output, 
                'Raw image' ,(100, 450), 
                fonte, 1.5, (255,0,0),2)  
    '''
    
 
    # cmd_vel theta definition
    theta = (320 - centroid_1[0])
    Vel = Twist()
    Vel.linear.x = 0.2
    Vel.angular.z = theta * 0.5
    if abs(Vel.angular.z) > 0.5: Vel.angular.z = np.sign(Vel.angular.z) * 0.5
    cmd_vel.publish(Vel)

    cv2.putText(cv_output, 
                str(theta) ,(100, 350), 
                fonte, 3, (0,0,255),2)  

    
    #cv_raw_red = cv2.resize(cv_raw, (W, H))
    M1 = cv2.resize(mask_show, (W, H))
    RGB_out = cv2.resize(cv_output, (W, H))

    M1 = cv2.resize(mask_show, (W, H))
    '''
    M1 = cv2.resize(mask1, (W, H))
    M2 = cv2.resize(mask2, (W, H))
    M3 = cv2.resize(mask_out, (W, H))
    '''
    output = np.vstack((M1,RGB_out))
    
    cv2.imshow('Robot view (CV Output)',output)
    #cv2.imshow('Robot view (CV Raw)', cv_raw)
    cv2.waitKey(2)      

#--------------------------------------------------------------------

def process_image_roi():
 
    # Image acquisition
    cv_output, I1_prime, I2_prime, I3_prime = get_img(0)
    #cv_raw = cv_output.copy()


    # Image resize parameters
    W = int(640*0.7)
    H = int(480*0.7)

    
    bk_h,bk_w,bk_d = cv_output.shape
    black_img = np.ones((bk_h,bk_w))*255

    rows, col, channels = cv_output.shape
    
    roi = cv_output[300:470,:,:]
    roi_I1 = I1_prime[300:470,:]
    roi_I2 = I2_prime[300:470,:]
    roi_I3 = I3_prime[300:470,:]

    # Image blurring 
    blur1 = cv2.GaussianBlur(roi_I1,(21,21),20)
    blur2 = cv2.GaussianBlur(roi_I3,(21,21),20)

    # Binary mask genteration 
    th,mask_out = cv2.threshold(blur1,
                            0,255,
                            cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    # Fiding mask contours
    contour = cv2.findContours(~mask_out, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contour = contour[1]
    
    # Contours parameters
    area_1 = 0
    area_2 = 0
    moment_1 = []
    moment_2 = []
    Cont_1 = []
    Cont_2 = []
    centroid_1 = [0,0]
    centroid_2 = [0,0]   

    # Fiding great area in the mask
    for c in contour:
        M = cv2.moments(c)
        if (M["m00"] > area_2):
            if (M["m00"] > area_1):
                area_2 = area_1
                moment_2 = moment_1
                moment_1 = M
                
                area_1 = M["m00"]
                Cont_2 = Cont_1
                Cont_1 = [c]
            else:
                area_2 = M["m00"]
                moment_2 = M
                Cont_2 = [c]

    
    if area_1 > 1000:
        centroid_1[0] = int(moment_1["m10"]/moment_1["m00"])
        centroid_1[1] = int(moment_1["m01"]/moment_1["m00"])

        cv2.circle(roi, (centroid_1[0], centroid_1[1]), 7, (255,0,0),-1)
        cv2.drawContours(roi, Cont_1 ,-1,(0,0,255),5)

        #cv2.circle(cv_output, (centroid_1[0], centroid_1[1]), 7, (255,0,0),-1)
        #cv2.drawContours(cv_output, Cont_1 ,-1,(0,255,0),2)
    #print Cont_1.size()

    K = Cont_1[0][0]
    s = max(K, key=lambda item: (-item[0], item[1]))
    print 'Area 1: ', area_1, ' Area 2: ', area_2, ' Max Contour elemts: ',s

    
    if centroid_1[0] > 320:
        l = centroid_1
        r = centroid_2  
        centroid_1 = r
        centroid_2 = l
    
    cv2.fillPoly(black_img, Cont_1, 255)
    
    cv_output[300:470,:,:] = roi
    mask_out_1 = np.zeros((W,H))
    fonte = cv2.FONT_HERSHEY_PLAIN
    
    cv2.putText(cv_output, 
                'Robot view' ,(100, 450), 
                fonte, 1.5, (0,0,255),2)    
    
    mask_show = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)
    
    cv2.putText(mask_show, 
                'Mask view' ,(100, 450), 
                fonte, 1.5, (0,255,0),2)    
    '''
    cv2.putText(cv_raw, 
                'Raw image' ,(100, 450), 
                fonte, 1.5, (255,0,0),2)  
    '''

    #cv_raw = cv2.resize(cv_raw, (W, H))
    M1 = cv2.resize(mask_show, (W, H))
    RGB_out = cv2.resize(cv_output, (W, H))
    output = np.vstack((M1, RGB_out))
    
    #i1_out = cv2.cvtColor(roi_I1, cv2.COLOR_GRAY2BGR)
    #i3_out = cv2.cvtColor(roi_I3, cv2.COLOR_GRAY2BGR)
    #output_roi = np.hstack((roi_out,i3_out))
    
    #cv2.imshow('ROI',i3_out)
    cv2.imshow('Robot view (Filtered)', mask_show)
    cv2.waitKey(2)      

#--------------------------------------------------------------------
