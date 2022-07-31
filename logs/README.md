(*) To log robot position:

1) Type this in new terminal window, while a2b_package is running: "rostopic echo -p /odom/pose/pose/position > ~/catkin_ws/src/a2b_controller/logs/buffer.csv"

2) In this directory always should be empty file "buffer.csv"

(*) To retrieve map logs:

1) go to $HOME/.ros/

2) move all Map_....png to this folder


