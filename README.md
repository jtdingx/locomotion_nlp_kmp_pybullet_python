# locomotion_nmpc_pybullet_python: bipedal and quadrupedal
=====using conda environment====
Build the pybullet simulation environment for legged robots' motion planning and control (coding by python.3.6/3.7/3.8)

main_loop: robot_nmpc_nlp_ik.py




# Dependency:


---The "Pinocchio" robotics dynamics library is used for robot IK computation

---NLP: adjust step location and step timing, solved by SDP using 'MOSEK' optimization library, you need an academic license
reference:
1,Ding, Jiatao, Xiaohui Xiao, and Nikos Tsagarakis. "Nonlinear optimization of step duration and step location." 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2019. 
2,Ding, Jiatao, et al. "Robust gait synthesis combining constrained optimization and imitation learning." 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020.


---KMP: natural leg trajectory generation by learning from demonstrations, see KMP.
reference:
3,Huang, Yanlong, et al. "Kernelized movement primitives." The International Journal of Robotics Research 38.7 (2019): 833-852.
