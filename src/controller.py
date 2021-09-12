#! /usr/bin/env python

import rospy
# importujeme metod "Twist" ktery bude posilat zpravy s tematem geometry_msgs
from geometry_msgs.msg import Twist



class TurtleBot:

    def __init__(self):
        rospy.init_node("speed_publisher")     # inicializace nodu s nazvem nodu
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 1) # definice publisheru, ktery bude posilat zpravy a turtlebot bude poslouchat tyto zpravy
        self.rate = rospy.Rate(2) # posilat zpravy s frekvenci 2 ms 


    def move2goal(self):
        goal_coord_arr = {"x": 0.0, "y":0.0}     # definice pole souradnic cile
        orig_coord_arr = {"x": 0.0, "y":0.0}     # definice pole pocatecnich souradnic

        # zadani souradnic cile
        goal_coord_arr["x"] = float(input("Souradnice cile x: "))
        goal_coord_arr["y"] = float(input("Souradnice cile y: "))

        vel_msg = Twist()   # definice objektu zpravy uvnitr ktereho budou linear a angular values - uhlove a linearni rychlosti turtlebotu
        
        i = 0

        while i<20:
            vel_msg.linear.x = 0.5 # pohyb podel osy x s rychlosti 0.5
            self.pub.publish(vel_msg)
            self.rate.sleep()
            i += 1

        # podminka dosazeni cile
        vel_msg.linear.x = 0.0 
        self.pub.publish(vel_msg)

        rospy.spin()    # ctrl+c zastavi node


while not rospy.is_shutdown():
    x = TurtleBot()
    x.move2goal()




