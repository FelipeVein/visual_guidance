#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import time
import cv2
from geometry_msgs.msg import Twist, Pose2D
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from math import atan2, sqrt, pi, hypot
import matplotlib.pyplot as plt
import imageProcess as imp


from copy import deepcopy


from filterpy.monte_carlo import residual_resample

import math



class ParticleFilter():
    def __init__(self, fitness, transition, nfeatures=5,nparticles=100, distribution=None, mu = None, sigma = None, lb = None, ub = None):
        self.fitness = fitness
        self.transition = transition
        self.nfeatures = nfeatures
        self.nparticles = nparticles
        self.distribution = distribution
        self.mu = mu
        self.sigma = sigma
        self.lb = lb
        self.ub = ub
        self.initialize_particles()
    

    def initialize_particles(self):
        self.particles = []
        if(self.distribution):
            for i in range(self.nfeatures):
                if(self.distribution[i] == 'normal'):
                    self.particles.append(np.random.normal(size=(self.nparticles, 1), loc = self.mu[i], scale=self.sigma[i]))
                else:
                    self.particles.append(np.random.uniform(low = self.lb[i], high = self.ub[i], size=(self.nparticles, 1)))
        else:
            self.particles = np.random.normal(size=(self.nparticles, self.nfeatures))
            '''if(self.sigma): 
                self.particles *= self.sigma
            if(self.mu):
                self.particles += self.mu'''
            
        self.particles = np.array(self.particles).reshape(self.nfeatures,self.nparticles)
        self.particles = np.swapaxes(self.particles, 0, 1)
        if(self.lb):
            self.particles = np.clip(self.particles, self.lb, np.inf)
        if(self.ub):
            self.particles = np.clip(self.particles, -np.inf, self.ub)
        self.weights = np.zeros(shape=(self.nparticles))

    
    def return_particles(self):
        return self.particles
    def return_20_mean(self):
        idx = np.argpartition(self.weights, int(self.nparticles * 0.8))

        return np.average(self.particles[idx[int(self.nparticles * 0.8):]], weights = self.weights[idx[int(self.nparticles * 0.8):]], axis = 0)
    def return_best_mean(self, value = 0.95):
        idx = np.argpartition(self.weights, int(self.nparticles * 0.95))
        #plt.plot(self.weights[idx[int(self.nparticles * 0.95):]])
        
        return np.average(self.particles[idx[int(self.nparticles * 0.95):]], weights = self.weights[idx[int(self.nparticles * 0.95):]], axis = 0)

    def return_mean(self):
        return np.mean(self.particles, axis= 0)

    def predict(self, u):
        self.particles = self.transition(self.particles, u)



    def update(self, observation):
        self.weights = self.fitness(self.particles, observation)
        self.weights += 1.e-20
        self.weights /= np.sum(self.weights)
    
    def resample(self):
        indexes = residual_resample(self.weights)
        self.particles = self.particles[indexes,:]
        self.weights = self.weights[indexes]

    


    def next_step(self, observation, u):
        self.predict(u)
        self.update(observation)
        self.resample()

        
# considering a particle = [h, l, rw, rd]

def transition(particles, u):
    dx = u[0]
    dh = u[1]
    particles[:,0] = np.clip(particles[:,0] + dh + np.random.normal(loc=0.0, scale=0.01,size=particles[:,0].shape), - 30 * np.pi/180, 30 * np.pi/180)
    particles[:,1] = particles[:,1] + dx * np.sin(particles[:,0]) + np.random.normal(loc=0.0, scale=0.1,size=particles[:,1].shape)
    particles[:,2] = np.clip(particles[:,2] + np.random.normal(loc=0.0, scale=10,size=particles[:,2].shape), 0, np.inf)
    particles[:,3] = np.clip(particles[:,3] + np.random.normal(loc=0.0, scale=10,size=particles[:,3].shape), 0, np.inf)
    return particles


def create_masks(particles, observation, plot = 0):
    masks = []
    shape = observation.shape
    value = 200
    shape = (shape[0], shape[1] + value)
    for particle in particles:
        aux = np.zeros(shape)
        central_x = shape[1]//2
        mean_l = central_x - particle[2]//2 + particle[1]
        mean_r = central_x + particle[2]//2 + particle[1]
        left_l = np.clip(mean_l - particle[3]//2, 0, shape[1])
        right_l = np.clip(mean_l + particle[3]//2, 0, shape[1])
        left_r = np.clip(mean_r - particle[3]//2, 0, shape[1])
        right_r = np.clip(mean_r + particle[3]//2, 0, shape[1])

        aux[:,int(left_l):int(right_l)] = 1
        aux[:,int(left_r):int(right_r)] = 1
        if(plot):
            aux = 1-aux

        aux = skew_image(aux, int(particle[0] * 4 * 180./np.pi))


        aux = aux[:, value/2:shape[1]-value/2]

        masks.append(aux)
    return np.array(masks)

def skew_image(image, value = 0):

    IMAGE_H = image.shape[0]
    IMAGE_W = image.shape[1]

    value = value * (IMAGE_W + IMAGE_H)/400.
    aa = 100./400*IMAGE_W + value 
    bb = 300./400*IMAGE_W + value 

    src = np.float32([[40./400 * IMAGE_W, 0], [360./400 * IMAGE_W, 0], [0, IMAGE_H], [IMAGE_W, IMAGE_H]])
    dst = np.float32([[aa, 0], [bb, 0], [0, IMAGE_H], [IMAGE_W, IMAGE_H]])

    M = cv2.getPerspectiveTransform(src, dst) 
    Minv = cv2.getPerspectiveTransform(dst, src) 

    warped_img = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H))
    
    return warped_img

def fitness(particles, observation):
    pred = create_masks(particles, observation)
    observation /= np.max(observation)
    observation = observation[None, :].astype(np.float32)
    intersection = observation * pred
    notObservation = 1 - observation
    union = observation + (notObservation * pred)

    result = (np.sum(intersection, axis=(-1,-2)) + 1.e-100) / (np.sum(union, axis=(-1,-2)) + 1.e-100)

    return result




def exg(im_to_binarize):
    im_to_binarize = im_to_binarize.astype(np.float)
    R_ = im_to_binarize[:,:,2]/np.max(im_to_binarize[:,:,2])
    G_ = im_to_binarize[:,:,1]/np.max(im_to_binarize[:,:,1])
    B_ = im_to_binarize[:,:,0]/np.max(im_to_binarize[:,:,0])
    
    r = R_/(R_+G_+B_)
    g = G_/(R_+G_+B_)
    b = B_/(R_+G_+B_)
    
    excess_red = 1.4*r - g
    excess_green = 2*g - r - b
    return excess_green


def exg_th(img, th = [0., 0.5]):
    a = exg(img)
    b = np.zeros(shape = a.shape)
    b[a<th[0]] = 0
    b[(a>=th[0]) & (a < th[1])] = (a[(a>=th[0]) & (a < th[1])] - th[0])/(th[1] - th[0])
    b[a >= th[1]] = 1
    return b
    








def actuate(theta, found_setpoint):
    global pub, Vel
    global Vmax, Vconst, Kp, Kpw, Wmax
    #Estado de controle autonomo utilizando controle de steering
    #if(found_setpoint == 1 and (theta > 0.15 or theta < -0.15)):
    if(found_setpoint == 1):
        Vel.linear.x = Vconst
        Vel.angular.z = theta * Kpw
        if abs(Vel.angular.z) > Wmax:
            Vel.angular.z = np.sign(Vel.angular.z) * Wmax
        pub.publish(Vel)
    elif found_setpoint == 0:
        Vel.linear.x = 0
        Vel.angular.z = 0
        pub.publish(Vel)
    else:
        Vel.linear.x = Vconst
        Vel.angular.z = 0
        pub.publish(Vel)



def image_callback(data):

    global im_rgb, bridge, flag

    im_rgb = cv2.cvtColor(bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")[100:, :], cv2.COLOR_BGR2RGB)
    #imagem = data.data
    flag = 1





def normalize(data):

    data_n = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    return data_n



def skew_depth(image, value = 0):

    IMAGE_H = image.shape[0]
    IMAGE_W = image.shape[1]

    aa = 200 + value
    bb = 450 + value

    src = np.float32([[200, IMAGE_H], [450, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[aa, IMAGE_H], [bb, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    M = cv2.getPerspectiveTransform(src, dst) 

    warped_img = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H)) 
    return warped_img

#theta = [(i-30) for i in range(61)]





def normalize(data):

    data_n = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    return data_n






def blend_images(rgb, mask):
    aux = cv2.merge([mask * 255, np.zeros(mask.shape), np.zeros(mask.shape)])
    aux = aux.astype(np.uint8)

    return cv2.addWeighted(rgb, 1, aux, 5, 0.0)

def draw_arrow(image, l, h, scale = 1):
    pt1 = (int(image.shape[1]/2 + l*scale), image.shape[0])
    pt2 = (int(image.shape[1]/2 + l*scale + image.shape[1]/2 * math.tan(h)), image.shape[0]/2)

    cv2.arrowedLine(image, pt1, pt2,(0,0,255), 4)
    return image


def blend_images_big(rgb,mask):
    aux = cv2.merge([mask * 255, np.zeros(mask.shape), np.zeros(mask.shape)])
    aux = cv2.resize(aux, (rgb.shape[1],rgb.shape[0]))
    aux = aux.astype(np.uint8)

    return cv2.addWeighted(rgb, 1, aux, 5, 0.0)



def run_control():
    global pub, Vel, imagem, im_rgb
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    Vel = Twist()

    process_image_size = (200,50)
    max_lateral_offset = process_image_size[0]/2
    max_row_width = process_image_size[0]
    max_row_distance = process_image_size[0]

    mean_row_width = process_image_size[0]/2
    mean_row_distance = process_image_size[0]/2
    global v, w
    v = 0
    w = 0

    nparticles = 100


    pf = ParticleFilter(transition = transition, fitness = fitness, nfeatures=4, 
        mu = [0,0,mean_row_width,mean_row_distance], sigma = [1,10,50,50], lb = [-np.pi/2, -max_lateral_offset, 1, 5], 
        ub=[np.pi/2, max_lateral_offset, max_row_width, max_row_distance], 
        nparticles = nparticles, distribution=['normal','normal','uniform','uniform'])

    while not rospy.is_shutdown():
       if(flag == 1):
            image_rs = cv2.resize(im_rgb,process_image_size)
            mask = exg_th(image_rs)
            
            pf.next_step(mask, [v,w]) # I dont know the input on each image, so I will let it be [0.1,0] for now

            best_particle = pf.return_best_mean()

            angle = best_particle[0]
            
            angle = -angle

            
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (0,100)
            fontScale              = 1
            fontColor              = (255,255,255)
            lineType               = 2

            mask_best_particle = create_masks([best_particle], mask, plot = 0)[0]

            blended_image = blend_images_big(im_rgb, mask_best_particle)

            cv2.putText(blended_image,str(angle), 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)


            #cv2.imshow('vector', vector)
            cv2.imshow('frame',blended_image)




            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(angle)




            actuate(angle,True)

            v = Vel.linear.x
            w = Vel.angular.z / Kpw
            


def main():
    global imagem, im_rgb, bridge, flag, Kp, Kpw, Vmax, Vconst, Wmax
    imagem = []
    im_rgb = []
    Kp = 0.5
    Kpw = 0.05
    Kpw = 5
    Vmax = 0.2
    Vconst = 0.5
    Wmax = 0.5
    flag = 0
    rospy.init_node('depth_control', anonymous = True)
    rospy.Subscriber("/soybot/center_camera1/image_raw", Image, image_callback)   
    bridge = CvBridge()
    run_control()

if __name__ == "__main__":
    main()

