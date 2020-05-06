

#!/usr/bin/env python
import rospy

from flexbe_core import EventState, Logger


import math
from math import atan2, sqrt, pi, hypot

from flexbe_core.proxy import ProxyPublisher
from flexbe_core.proxy import ProxySubscriberCached

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

from sensor_msgs.msg import Image


import matplotlib.pyplot as plt

import numpy as np

import cv2

from copy import deepcopy

import imageProcess as imp
from cv_bridge import CvBridge, CvBridgeError


class DepthControl(EventState):
    '''
    Example for a state to demonstrate which functionality is available for state implementation.
    This example lets the behavior wait until the given target_time has passed since the behavior has been started.

    -- target_time 	float 	Time which needs to have passed since the behavior started.

    <= continue 			Given time has passed.
    <= failed 				Example for a failure outcome.

    '''

    def __init__(self, rate):
        # Declare outcomes, input_keys, and output_keys by calling the super constructor with the corresponding arguments.
        super(DepthControl, self).__init__(outcomes = ['finished', 'failed'])
        self._failed = False
        
        self.Kpw = 0.1
        self.Vmax = 0.2
        self.Vconst = 0.5
        self.Wmax = 0.5

        self.set_rate(rate) ## change the rate of 'execute'

        self.cmd_vel = '/cmd_vel'
        self.depth = '/soybot/center_camera/depth/image_raw'
        self._sub = ProxySubscriberCached({self.depth: Image})
        self._pub = ProxyPublisher({self.cmd_vel: Twist})

        self.bridge = CvBridge()



        ## so da para mudar rate por meio do GUI. Podemos pegar rosparam quando for rodar sem GUI. 


    def normalize(self,data):

        data_n = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
        return data_n

    def preprocessing(self,imagem):
        imagem_corrigida = deepcopy(imagem)

        imagem_corrigida.setflags(write=1)
        where_are_NaNs = np.isnan(imagem_corrigida)
        imagem_corrigida[where_are_NaNs] = 3.0


        img = self.normalize(imagem_corrigida).astype(np.uint8)
        yf, xf = img.shape
        x = 150
        img = img[200:yf, x:xf-x]
        #img = cv2.medianBlur(img,15)
        imga = deepcopy(img)
        img = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(1,640)).apply(img)
        
        #imagempp = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,10)
        
        _, imagempp = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        a = np.sum(~imagempp[0:imagempp.shape[0]//2, :])

        #Logger.log('{}'.format(a), Logger.REPORT_HINT)

        if(a <= 2052240):# 4855200 para //5
            end_row = 1
        else:
            end_row = 0

        return img, imagempp, imga, end_row

    def actuate(self,theta, found_setpoint):
        Vel = Twist()
        if(found_setpoint == 1):
            Vel.linear.x = self.Vconst
            Vel.angular.z = theta * self.Kpw
            if abs(Vel.angular.z) > self.Wmax:
                Vel.angular.z = np.sign(Vel.angular.z) * self.Wmax
        else:
            Vel.linear.x = 0
            Vel.angular.z = 0
        try:
            self._pub.publish(self.cmd_vel, Vel)
        
        except Exception as e:
            Logger.logwarn('Failed to send velocity:\n%s' % str(e))
            self._failed = True


    def execute(self, userdata):
        # This method is called periodically while the state is active.
        # Main purpose is to check state conditions and trigger a corresponding outcome.
        # If no outcome is returned, the state will stay active.


        if self._sub.has_msg(self.depth):
            data = self._sub.get_last_msg(self.depth)
            depth = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            _, imagempp, _, end_row = self.preprocessing(depth)
            if(end_row):
                self.actuate(0, 0)
                return 'finished'

            frame, mask_show, centroid_1, mask_show2 = imp.get_centroid(~imagempp, ~imagempp, ~imagempp, ~imagempp)

	
            y,x = imagempp.shape

            if(centroid_1[0] > 0):
                  theta = (x/2 - centroid_1[0])
                  theta = round(atan2(theta,y/2),2)
                  found_setpoint = 1
            else:
                    found_setpoint = 0
            self.actuate(theta,found_setpoint)

            
        if(self._failed):
            return 'failed'
            

        
        

    def on_enter(self, userdata):
        # This method is called when the state becomes active, i.e. a transition from another state to this one is taken.
        # It is primarily used to start actions which are associated with this state.

        # The following code is just for illustrating how the behavior logger works.
        # Text logged by the behavior logger is sent to the operator and displayed in the GUI.
        self.failed = False
        

    def on_exit(self, userdata):
        # This method is called when an outcome is returned and another state gets active.
        # It can be used to stop possibly running processes started by on_enter.

        pass # Nothing to do in this example.


    def on_start(self):
        # This method is called when the behavior is started.
        # If possible, it is generally better to initialize used resources in the constructor
        # because if anything failed, the behavior would not even be started.

        # In this example, we use this event to set the correct start time.
        #self._start_time = rospy.Time.now()
        pass


    def on_stop(self):
        # This method is called whenever the behavior stops execution, also if it is cancelled.
        # Use this event to clean up things like claimed resources.

        pass # Nothing to do in this example.
        
