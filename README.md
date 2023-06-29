# locomotion_nmpc_pybullet_python： bipedal and quadrupedal

conda environment: 
Build the pybullet simulation environment for legged robots' motion planning and control (coding by python.3.6/3.7/3.8)

main_loop: robot_nmpc_nlp_ik.py



dependency:
The Pinocchio dynamic library is used for robot IK computation

NLP: adjust step location and step timing, solved by SDP using MOSEK optimization librar

KMP: natural leg trajectory generation by learning from demo.
