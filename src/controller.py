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

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)  # definice publisheru,
        # ktery bude posilat zpravy a turtlebot
        # bude poslouchat zpravy

        self.sub = rospy.Subscriber("/scan", LaserScan, callback)  # definice subscriberu, ktery bude sledovat
        # tema /scan, tema s kterou publikuji
        # senzory turtlebotu

        self.rate = rospy.Rate(2)  # posilat zpravy s frekvenci 2 ms

        self.safety_boundary = 0.4
        self.lin_vel = 0.1
        self.ang_vel = 0.2

        self.safe_dx = 0.2
        self.variance_of_distance = 0.2

    @staticmethod  # does not use self in it
    def goal_angle(goal_coord_arr, orig_coord_arr):
        return atan2(goal_coord_arr["x"] - orig_coord_arr["x"], goal_coord_arr["y"] - orig_coord_arr["y"])

    @staticmethod
    def goal_distance(goal_coord_arr, orig_coord_arr):
        return sqrt(
            pow((goal_coord_arr["x"] - orig_coord_arr["x"]), 2) + pow((goal_coord_arr["y"] - orig_coord_arr["y"]), 2))

    def rotation_move(self, angle, ang_vel):

        print("Moving angular with", ang_vel, "speed")  # debug

        vel_msg = Twist()  # definice objektu zpravy uvnitr ktereho budou linear a angular values -
        # uhlove a linearni rychlosti turtlebotu

        t0 = rospy.Time.now().to_sec()  # pro zaznamenani casu
        t1 = t0

        while ang_vel * (t1 - t0) < abs(angle):

            #################
            ## NAIVE STOP! ##
            #################
            if rospy.is_shutdown():
                self.stop_tbot()
                self.rate.sleep()
                return None
            #################
            ## NAIVE STOP! ##
            #################

            if angle <= 0:
                vel_msg.angular.z = -ang_vel
            else:
                vel_msg.angular.z = ang_vel
            self.pub.publish(vel_msg)
            t1 = rospy.Time.now().to_sec()
            self.rate.sleep()

        vel_msg.angular.z = 0.0
        self.pub.publish(vel_msg)
        self.rate.sleep()

        return None

    def linear_move(self, distance, lin_vel):

        print("Moving linear with", lin_vel, "speed") # debug

        obstacle_avoided: bool = False
        curr_length_of_path: float = 0
        length_of_obstacle: float = 0

        vel_msg = Twist()  # definice objektu zpravy uvnitr ktereho budou linear a angular values - uhlove a linearni
        # rychlosti turtlebotu

        t0 = rospy.Time.now().to_sec()  # pro zaznamenani casu
        t1 = t0

        while (lin_vel * (t1 - t0) < distance):

            #################
            ## NAIVE STOP! ##
            #################
            if rospy.is_shutdown():
                self.stop_tbot()
                self.rate.sleep()
                return None
            #################
            ## NAIVE STOP! ##
            #################

            vel_msg.linear.x = lin_vel  # pohyb podel osy x robota s rychlosti 0.5
            self.pub.publish(vel_msg)
            self.rate.sleep()
            t1 = rospy.Time.now().to_sec()
            # podminka prekazky
            if distance_obst_arr["0 deg"] < self.safety_boundary:
                print("obstacle detected in ", distance_obst_arr["0 deg"])
                curr_length_of_path = lin_vel*(t1-t0)
                obstacle_avoided, _ = self.naive_obstacle_avoidance()
                while not obstacle_avoided:
                    obstacle_avoided, length_of_obstacle = self.naive_obstacle_avoidance()
                break

        # podminka dosazeni cile
        vel_msg.linear.x = 0.0
        self.pub.publish(vel_msg)

        # dojet cil
        if obstacle_avoided:
            self.linear_move(distance - curr_length_of_path - length_of_obstacle, self.lin_vel)

        return None

    def stop_tbot(self):

        print("stop tbot")

        vel_msg = Twist()  # definice objektu zpravy uvnitr ktereho budou linear a angular values - uhlove a linearni
        # rychlosti turtlebotu

        vel_msg.linear.x = 0

        vel_msg.angular.z = 0

        self.pub.publish(vel_msg)
        self.rate.sleep()

        return None

    def naive_obstacle_avoidance(self):

        print("naive obstacle avoidance")

        #################
        ## NAIVE STOP! ##
        #################
        if rospy.is_shutdown():
            self.stop_tbot()
            self.rate.sleep()
            return None
        #################
        ## NAIVE STOP! ##
        #################

        obstacle_avoided: bool = False  # Flag, that obstacle is avoided
        deviation_from_x: float = 0
        deviation_from_y: float = 0
        residue_of_avoidance = 0.9     # error of avoidance due to odometry (very big error!!)

        vel_msg = Twist()  # definice objektu zpravy uvnitr ktereho budou linear a angular values - uhlove a linearni
        # rychlosti turtlebotu

        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        self.pub.publish(vel_msg)

        # otoceni vlevo
        self.rotation_move(pi/2, self.ang_vel)

        # jizda pokud neprejede prekazku (pokud je v intervalu bezpecne distance)
        while (distance_obst_arr["270 deg"] <= (self.safety_boundary + self.variance_of_distance)) \
                and (distance_obst_arr["270 deg"] >= (self.safety_boundary - self.variance_of_distance)):
            print("first distance with safe margin, ", distance_obst_arr["270 deg"])
            self.linear_move(self.safe_dx, self.lin_vel)
            deviation_from_y += self.safe_dx

        # otoceni vlevo
        self.rotation_move(-pi / 2, self.ang_vel)

        # jizda pokud zase nedojede do prekazky (mimo interval bezpeci)
        while distance_obst_arr["270 deg"] > (self.safety_boundary + self.variance_of_distance):
            print("second distance with safe margin, ", distance_obst_arr["270 deg"])
            self.linear_move(self.safe_dx, self.lin_vel)

        # jizda pokud zase neprejede prekazku (pokud je v intervalu bezpecne distance)
        while (distance_obst_arr["270 deg"] <= (self.safety_boundary + self.variance_of_distance)) \
                and (distance_obst_arr["270 deg"] >= (self.safety_boundary - self.variance_of_distance)):
            print("third distance with safe margin, ", distance_obst_arr["270 deg"])
            self.linear_move(self.safe_dx, self.lin_vel)
            deviation_from_x += self.safe_dx

        # otoceni vpravo
        self.rotation_move(-pi / 2, self.ang_vel)

        # jizda zpadky na cestu
        print("forth distance with safe margin, ", distance_obst_arr["270 deg"])
        self.linear_move(deviation_from_y, self.lin_vel)

        # otoceni na spravny smer
        self.rotation_move(pi / 2, self.ang_vel)

        obstacle_avoided = True

        return obstacle_avoided, deviation_from_x

    def move2goal(self):

        print("move2goal")

        #################
        ## NAIVE STOP! ##
        #################
        if rospy.is_shutdown():
            self.stop_tbot()
            self.rate.sleep()
            return None
        #################
        ## NAIVE STOP! ##
        #################

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


def main():

    rospy.init_node("controller_py_n")  # inicializace nodu s nazvem nodu

    tb = TurtleBot()
    tb.move2goal()
    if rospy.is_shutdown():
        tb.stop_tbot()
    #rospy.spin()


if __name__ == '__main__':
    main()
