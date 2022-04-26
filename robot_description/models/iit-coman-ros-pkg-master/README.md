iit-coman-ros-pkg
=================

Repository for the compliant humanoid robot COMAN from the Italian Institute of Technology contains a set of ROS packages. The updated packages are ```coman_urdf```, ```coman_gazebo``` and ```coman_srdf``` while the others are no longer updated.

This repository can be used with ROS with a limited set of functionalities. 

#Package list:

coman_urdf: 
-----------
contains the urdf.xacro description of COMAN as well as files that are needed in the simulation in GAZEBO. 
Every time one of these files are changed you have to use the script inside the script/ folder to generate. The script generates files that are used in simulation as well as files that can be used inside kinematics/dynamics libraries (coman.urdf). Running this script requires ROS and some python dependencies. You can just install a precompiled version of the urdf and sdf, though. Instructions are in the ```coman_gazebo``` section of this readme.

After installing ros (tested on ros hydro),  make sure the dependency on BeautifulSoup4 is satisfied before using the script:
```
sudo easy_install beautifulsoup4
```
If you have problems with this command maybe you have to install ```python``` or the package ```easy_install```.
After this, you have to be sure that ```coman_urdf``` is inside your ros path.
Either put it in your ros package path or do something like this:
```
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/your_name/projects/walkman/iit-coman-ros-pkg
```

The script has to be called in script/ folder:
```
./create_urdf_and_sdf.sh coman
```
It generates the coman urdf and srdf descriptions and sdf model as specified in ```coman_config.urdf.xacro```.

After running the script you can check the results directly by running
```
roslaunch coman_urdf coman_slider.launch
rosrun rviz rviz
```
or directly in simulation.


coman_gazebo:
-------------
contains files needed for the simulation as well as a launch file to start the simulation.
The coman_gazebo folder includes precompiled sdf and urdf files for coman, as well as many initialization files for the setting of the control parameters for the simulated robot (YARP VERSION).
To install these models files, you should first remove the version you already have installed (make sure to make a backup if you manually modified the configuration files)
```
rm -r ~/.gazebo/model/coman_urdf
```
And to install the new version, from the coman_gazebo folder, run
```
mkdir ~/.gazebo/models/coman_urdf
cp -R sdf/* ~/.gazebo/models/coman_urdf/
```

coman_srdf: 
-----------
contains the kinematic description of COMAN intended as number of kinematich chains and joints included in each chain. These file is automatically generated by the script considering the ```coman_config.urdf.xacro```.


coman_control (NO LONGER MANTAINED):
------------------------------------
contains the files needed to control the robot. For now the available controller is the effort_controllers/JointPositionController. 
To start the controller:
```
roslaunch coman_control coman_control.launch
```
It is also implemented a node that permits to send trajectory to the robot that were previously recorded in a file, 
to run this node:
```
rosrun coman_control send_trajectories
```
in another terminal runs:
```
rosrun rqt_reconfigure rqt_reconfigure
```
remember to run this command in the same folder were is the traj.dat file. The format of this file can be found in coman_control/src.

coman_state_publisher (NO LONGER MANTAINED):
--------------------------------------------
contains a node that permits to publish inside ROS the state of the real robot. Communication with COMAN occurs 
through xddp_sockets. This package has an external dependency against Robolli libraries. 
We are working to make it independent. 

coman_msgs (NO LONGER MANTAINED): 
---------------------------------
contains msgs that are used in coman_state_publisher.

How to cite this work:
======================
Please support ```iit-coman-ros-pkg``` development by referencing it in your works/publications/projects with:
```
@Inbook{MingoHoffman2014,
  author="Mingo Hoffman, Enrico and Traversaro, Silvio and Rocchi, Alessio and Ferrati, Mirko and Settimi, Alessandro and Romano, Francesco and Natale, Lorenzo and Bicchi, Antonio and Nori, Francesco and Tsagarakis, Nikos G.",
  editor="Hodicky, Jan",
  title="Yarp Based Plugins for Gazebo Simulator",
  bookTitle="Modelling and Simulation for Autonomous Systems: First International Workshop, MESAS 2014, Rome, Italy, May 5-6, 2014, Revised Selected Papers",
  year="2014",
  publisher="Springer International Publishing",
  address="Cham",
  pages="333--346",
  isbn="978-3-319-13823-7",
  doi="10.1007/978-3-319-13823-7_29",
  url="http://dx.doi.org/10.1007/978-3-319-13823-7_29"
}
```
