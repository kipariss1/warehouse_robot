#! /usr/bin/env python

import rospy
# importujeme metod "Twist" ktery bude posilat zpravy s tematem geometry_msgs
from geometry_msgs.msg import Twist
from math import atan2, pow, sqrt, pi
# importujeme metod "Laser scan" ktery bude koukat na hodnoty ze senzoru
from sensor_msgs.msg import LaserScan
from typing import Any, Dict


def callback(msg):
    global distance_obst_arr
    # globalni dictionary do ktereho se zapisuji hodnoty
    # ze sensoru, kolik zbylo do prekazky pos uhlem 0, 90
    distance_obst_arr = {"0 deg": msg.ranges[0], "90 deg": msg.ranges[90], "180 deg": msg.ranges[180],
                         "270 deg": msg.ranges[270]}


class TurtleBot:

    def __init__(self):
        rospy.init_node("speed_publisher")  # inicializace nodu s nazvem nodu

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)  # definice publisheru,
        # ktery bude posilat zpravy a turtlebot
        # bude poslouchat zpravy

        self.sub = rospy.Subscriber("/scan", LaserScan, callback)  # definice subscriberu, ktery bude sledovat
        # tema /scan, tema s kterou publikuji
        # senzory turtlebotu

        self.rate = rospy.Rate(2)  # posilat zpravy s frekvenci 2 ms

        self.safety_boundary = 0.5
        self.lin_vel = 0.1
        self.ang_vel = 0.1

    @staticmethod  # does not use self in it
    def goal_angle(goal_coord_arr, orig_coord_arr):
        return atan2(goal_coord_arr["x"] - orig_coord_arr["x"], goal_coord_arr["y"] - orig_coord_arr["y"])

    @staticmethod
    def goal_distance(goal_coord_arr, orig_coord_arr):
        return sqrt(
            pow((goal_coord_arr["x"] - orig_coord_arr["x"]), 2) + pow((goal_coord_arr["y"] - orig_coord_arr["y"]), 2))

    def rotation_move(self, angle, ang_vel):

        vel_msg = Twist()  # definice objektu zpravy uvnitr ktereho budou linear a angular values -
        # uhlove a linearni rychlosti turtlebotu

        t0 = rospy.Time.now().to_sec()  # pro zaznamenani casu
        t1 = t0

        while ang_vel * (t1 - t0) < abs(angle):
            if angle <= 0:
                vel_msg.angular.z = -ang_vel
            else:
                vel_msg.angular.z = ang_vel
                self.pub.publish(vel_msg)
                t1 = rospy.Time.now().to_sec()

            vel_msg.angular.z = 0.0
            self.pub.publish(vel_msg)

            rospy.spin()  # ctrl+c zastavi node

    def linear_move(self, distance, lin_vel):

        vel_msg = Twist()  # definice objektu zpravy uvnitr ktereho budou linear a angular values - uhlove a linearni
        # rychlosti turtlebotu

        t0 = rospy.Time.now().to_sec()  # pro zaznamenani casu
        t1 = t0

        while (lin_vel * (t1 - t0) < distance):
            vel_msg.linear.x = lin_vel  # pohyb podel osy x robota s rychlosti 0.5
            self.pub.publish(vel_msg)
            self.rate.sleep()
            t1 = rospy.Time.now().to_sec()
            # podminka prekazky
            if distance_obst_arr["0 deg"] < 0.5:
                self.naive_obstacle_avoidance()
                while not obstacle_avoided:
                    obstacle_avoided = self.naive_obstacle_avoidance()

        # podminka dosazeni cile
        vel_msg.linear.x = 0.0
        self.pub.publish(vel_msg)

        rospy.spin()  # ctrl+c zastavi node

    def naive_obstacle_avoidance(self):
        obstacle_avoided: bool = False  # Flag, that obstacle is avoided

        vel_msg = Twist()  # definice objektu zpravy uvnitr ktereho budou linear a angular values - uhlove a linearni
        # rychlosti turtlebotu

        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        self.pub.publish(vel_msg)

        self.rotation_move(pi / 2, 0.1)

        rospy.spin()  # ctrl+c zastavi node

        return obstacle_avoided

    def move2goal(self):
        goal_coord_arr: Dict[str, Any] = {"x": 0.0, "y": 0.0}  # definice pole souradnic cile
        orig_coord_arr: Dict[str, Any] = {"x": 0.0, "y": 0.0}  # definice pole pocatecnich souradnic

        # zadani souradnic cile
        goal_coord_arr["x"] = float(input("Souradnice cile x: "))
        goal_coord_arr["y"] = float(input("Souradnice cile y: "))

        ####OTACENI####
        angle = self.goal_angle(goal_coord_arr, orig_coord_arr)

        self.rotation_move(angle, self.ang_vel)
        ###############

        #####JIZDA####
        distance = self.goal_distance(goal_coord_arr, orig_coord_arr)

        self.linear_move(distance, self.lin_vel)
        ##############

        rospy.spin()  # ctrl+c zastavi node


while not rospy.is_shutdown():
    x = TurtleBot()
    x.move2goal()
