#! /usr/bin/env python

import rospy
# importujeme metod "Twist" ktery bude posilat zpravy s tematem geometry_msgs
from geometry_msgs.msg import Twist




rospy.init_node("speed_publisher")     # inicializace nodu s nazvem nodu

# definice publisheru, ktery bude posilat zpravy a turtlebot bude poslouchat tyto zpravy

pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)

rate = rospy.Rate(2)    # posilat zpravy s frekvenci 2 ms

# definice objektu zpravy uvnitr ktereho budou linear a angular values - uhlove a linearni rychlosti turtlebotu
move = Twist()

move.linear.x = 0.5 # pohyb podel osy x s rychlosti 0.5

while not rospy.is_shutdown():
    pub.publish(move)
    rate.sleep()



