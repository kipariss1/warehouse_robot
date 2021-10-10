#! /usr/bin/env python

import rospy
# importujeme metod "Twist" ktery bude posilat zpravy s tematem geometry_msgs
from geometry_msgs.msg import Twist
from math import atan2, pow, sqrt
# importujeme metod "Laser scan" ktery bude koukat na hodnoty ze senzoru
from sensor_msgs.msg import LaserScan




def callback(msg):
	global distance_obst_arr
	distance_obst_arr = {"0 deg":msg.ranges[0]}		# globalni dictionary do ktereho se zapisuji hodnoty ze sensoru, kolik zbylo do prekazky pos uhlem 0


class TurtleBot:

    def __init__(self):
        rospy.init_node("speed_publisher")     # inicializace nodu s nazvem nodu
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 1) 	# definice publisheru, ktery bude posilat zpravy a turtlebot bude poslouchat tyto 											# zpravy
        
        self.sub = rospy.Subscriber("/scan", LaserScan, callback)		# definice subscriberu, ktery bude sledovat tema /scan, tema s kterou publikuji
        									# senzory turtlebotu
        self.rate = rospy.Rate(2) # posilat zpravy s frekvenci 2 ms 


    def goal_angle(self, goal_coord_arr, orig_coord_arr):
        return atan2(goal_coord_arr["x"] - orig_coord_arr["x"], goal_coord_arr["y"] - orig_coord_arr["y"])


    def goal_distance(self, goal_coord_arr, orig_coord_arr):
        return sqrt(pow((goal_coord_arr["x"] - orig_coord_arr["x"]), 2) + pow((goal_coord_arr["y"] - orig_coord_arr["y"]), 2))
        
        
    def naive_obstacle_avoidance(self, dist_from_orig):
    
    	# TODO vypocet novych sour 
    	
    	pass    	


    def move2goal(self):
        goal_coord_arr = {"x": 0.0, "y":0.0}     # definice pole souradnic cile
        orig_coord_arr = {"x": 0.0, "y":0.0}     # definice pole pocatecnich souradnic

        # zadani souradnic cile
        goal_coord_arr["x"] = float(input("Souradnice cile x: "))
        goal_coord_arr["y"] = float(input("Souradnice cile y: "))

        vel_msg = Twist()   # definice objektu zpravy uvnitr ktereho budou linear a angular values - uhlove a linearni rychlosti turtlebotu
        

        ####OTACENI####

        angle = self.goal_angle(goal_coord_arr, orig_coord_arr)
        t0 = rospy.Time.now().to_sec()   # pro zaznamenani casu
        t1 = t0
        ang_vel = 0.1                    # uhlova rychlos

        while (ang_vel*(t1-t0)<abs(angle)):
            if (angle<=0):
                vel_msg.angular.z = -ang_vel
            else:
                vel_msg.angular.z = ang_vel
            self.pub.publish(vel_msg)
            t1 =  rospy.Time.now().to_sec()

        vel_msg.angular.z = 0.0
        self.pub.publish(vel_msg)

        ###############

        #####JIZDA####

        distance = self.goal_distance(goal_coord_arr, orig_coord_arr)
        t0 = rospy.Time.now().to_sec()   # pro zaznamenani casu
        t1 = t0
        lin_vel = 0.1                    # linearni rychlos

        while (lin_vel*(t1-t0)<distance):
            vel_msg.linear.x = lin_vel # pohyb podel osy x robota s rychlosti 0.5
            self.pub.publish(vel_msg)
            self.rate.sleep()
            t1 = rospy.Time.now().to_sec()    
            if (distance_obst_arr["0 deg"]<0.8):
            	print("I've noticed ya!")		# TODO: implement here function naive_obstacle_avoidance            	

        # podminka dosazeni cile
        vel_msg.linear.x = 0.0 
        self.pub.publish(vel_msg)

        ##############

        rospy.spin()    # ctrl+c zastavi node


while not rospy.is_shutdown():
    x = TurtleBot()
    x.move2goal()




