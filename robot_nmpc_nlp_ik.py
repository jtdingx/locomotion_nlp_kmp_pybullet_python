#### python envir setup
from __future__ import print_function

import os
from os.path import dirname, join, abspath
import sys

from pathlib import Path


### pinocchio
import pinocchio as pin
from pinocchio.explog import log
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from pino_robot_ik import CLIK                  #### IK solver
from robot_tracking_controller import Gait_Controller #### controller

##### numpy
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
from numpy.linalg import norm, solve


##### pybullet
import pybullet
import pybullet_data
from sim_env import SimEnv
from sim_robot import SimRobot
import time
import math
import datetime

##### subprocess for external .exe
import subprocess

import scipy

import string

import copy
import mosek

from functools import wraps
from KMP_class import KMP
from NLP_sdp_class import NLP
import matplotlib.pyplot as plt


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.func_name, str(t1 - t0))
              )
        return result

    return function_timer





############################################## main loop function definition ###########################################
################################ pinocchio urdf setup ##################################
def addFreeFlyerJointLimits(robot):
    rmodel = robot.model

    ub = rmodel.upperPositionLimit
    ub[:7] = 1e-6
    rmodel.upperPositionLimit = ub
    lb = rmodel.lowerPositionLimit
    lb[:7] = -1e-6
    rmodel.lowerPositionLimit = lb

############################ NMPC c++ run for gait generation ##########################################################
#convert string  to number
def str2num(LineString, comment='#'):
    from io import StringIO as StringIO
    import re, numpy

    NumArray = numpy.empty([0], numpy.int16)
    NumStr = LineString.strip()
    # ~ ignore comment string
    for cmt in comment:
        CmtRe = cmt + '.*$'
        NumStr = re.sub(CmtRe, " ", NumStr.strip(), count=0, flags=re.IGNORECASE)

    # ~ delete all non-number characters,replaced by blankspace.
    NumStr = re.sub('[^0-9.e+-]', " ", NumStr, count=0, flags=re.IGNORECASE)

    # ~ Remove incorrect combining-characters for double type.
    NumStr = re.sub('[.e+-](?=\s)', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)
    NumStr = re.sub('[.e+-](?=\s)', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)
    NumStr = re.sub('[e+-]$', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)
    NumStr = re.sub('[e+-]$', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)

    if len(NumStr.strip()) > 0:
        StrIOds = StringIO(NumStr.strip())
        NumArray = numpy.genfromtxt(StrIOds)

    return NumArray

def run_nmpc_external_ext(j,cpptest):
    b = str(j)
    # if os.path.exists(cpptest):
    rc, out = subprocess.getstatusoutput(cpptest + ' ' + b)
    donser = str2num(out)

    return donser

############## IK computing
def joint_lower_leg_ik(robot,oMdesl,JOINT_IDl,oMdesr,JOINT_IDr,Freebase,Homing_pose,t,t_homing,robotname):
    ############ IK-solution ###############################################################33
    IK_left_leg = CLIK(robot, oMdesl, JOINT_IDl, Freebase)


    q1 = copy.deepcopy(robot.q0)
    if t < t_homing + 0.05:
        if Freebase:
            q1[0 + 7:6 + 7] = Homing_pose[-12:-6]
            q1[6 + 7:12 + 7] = Homing_pose[-6:]
        else:
            q1 = Homing_pose

    ############### Jacobian-based IK
    ql, Jl = IK_left_leg.ik_Jacobian(q=q1, Freebase=Freebase, eps=1e-8, IT_MAX=1000, DT=1e-1, damp=1e-6)

    q = copy.deepcopy(robot.q0)
    if t < t_homing + 0.05:
        if Freebase:
            q[0 + 7:6 + 7] = Homing_pose[-12:-6]
            q[6 + 7:12 + 7] = Homing_pose[-6:]
        else:
            q = Homing_pose
    IK_right_leg = CLIK(robot, oMdesr, JOINT_IDr, Freebase)
    qr, Jr = IK_right_leg.ik_Jacobian(q=q, Freebase=Freebase, eps=1e-8, IT_MAX=1000, DT=1e-1, damp=1e-6)
    # print("ql:", ql)
    # print("qr:", qr)
    ############################################################
    #############################################################

    ##### transfer the pinocchio joint to pybullet joint##########################
    q_ik = Homing_pose
    if (robotname == 'Talos'):
        if Freebase:
            q_ik[-12:-6] = ql[0 + 7:6 + 7]
            q_ik[-6:] = qr[6 + 7:12 + 7]

        else:
            q_ik[-12:-6] = ql[0:6]
            q_ik[-6:] = qr[6:12]
    else:
        if Freebase:
            q_ik[-12:-6] = ql[6 + 7:12 + 7]
            q_ik[-6:] = qr[6 + 7:12 + 7]

        else:
            q_ik[-12:-6] = ql[6:12]
            q_ik[-6:] = qr[6:12]
    return q_ik



def RotMatrixfromEulerx(xyz):
    x_angle = xyz[0]
    y_angle = xyz[1]
    z_angle = xyz[2]
    Rrpy = np.array([[math.cos(y_angle) * math.cos(z_angle), math.cos(z_angle) * math.sin(x_angle) * \
                      math.sin(y_angle) - math.cos(x_angle) * math.sin(z_angle),
                      math.sin(x_angle) * math.sin(z_angle) + \
                      math.cos(x_angle) * math.cos(z_angle) * math.sin(y_angle)],
                     [math.cos(y_angle) * math.sin(z_angle), math.cos(x_angle) * math.cos(z_angle) + \
                      math.sin(x_angle) * math.sin(y_angle) * math.sin(z_angle),
                      math.cos(x_angle) * math.sin(y_angle) * math.sin(z_angle) \
                      - math.cos(z_angle) * math.sin(x_angle)],
                     [-math.sin(y_angle), math.cos(y_angle) * math.sin(x_angle),
                      math.cos(x_angle) * math.cos(y_angle)]])
    # qua_base = pybullet.getQuaternionFromEuler(xyz)
    # Rrpy = pybullet.getMatrixFromQuaternion(qua_base)

    return Rrpy



def main():
    #################################################################################################
    global base_homing
    global pr_homing
    global pl_homing

    ############################################################################### coman robot simulation ######################################################################
    # ############################################################################# robot setup ###############
    robotname = 'Coman'  ### 'Talos','Coman','Cogimon' ################### For coman robot
    ########whold-body simu:
    Full_body_simu = True
    ##########for robot with float-base: humanoids or ################################
    Freebase = True

    ######### Coman: with/without forearm #####
    Forearm = False

    mesh_dirx = str(Path(__file__).parent.absolute())
    print(mesh_dirx)
    mesh_dir = mesh_dirx + '/robot_description/models/'

    # You should change here to set up your own URDF file
    if Full_body_simu:
        if (robotname == 'Talos'):
            urdf_filename = mesh_dir + '/talos_description/urdf/talos_full_no_grippers.urdf'
        elif ((robotname == 'Cogimon')):
            urdf_filename = mesh_dir + 'iit-cogimon/model.urdf'
        else:
            if Forearm:
                urdf_filename = mesh_dir + 'iit-coman/model.urdf'
            else:
                # urdf_filename = mesh_dir + 'iit-coman-no-forearms/model_jiatao.urdf'
                urdf_filename = mesh_dir + 'iit-coman-no-forearms/model.urdf'

    else:
        if (robotname == 'Talos'):
            urdf_filename = mesh_dir + 'talos_description/urdf/talos_lower_body_mesh_updated.urdf'
        elif ((robotname == 'Cogimon')):
            urdf_filename = mesh_dir + 'iit-cogimon-pennacchio/model.urdf'
        else:
            urdf_filename = mesh_dir + 'iit-coman-lowerbody-only/model.urdf'

    ### pinocchio load urdf
    print(urdf_filename)
    print(mesh_dir)
    if Freebase:
        robot = RobotWrapper.BuildFromURDF(urdf_filename, mesh_dir, pin.JointModelFreeFlyer())
        addFreeFlyerJointLimits(robot)
    else:
        robot = RobotWrapper.BuildFromURDF(urdf_filename, mesh_dir)
    # ## explore the model class
    # for name, function in robot.model.__class__.__dict__.items():
    #     print(' **** %s: %s' % (name, function.__doc__))
    # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444' )
    # print('standard model: dim=' + str(len(robot.model.joints)))
    # for jn in robot.model.joints:
    #     print(jn)
    # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$' )

    # find lower-leg joint idx in pinocchio
    if (robotname == 'Talos'):
        joints_desired_l = ['leg_left_1_joint', 'leg_left_2_joint', 'leg_left_3_joint', 'leg_left_4_joint',
                            'leg_left_5_joint', 'leg_left_6_joint']
        joints_desired_r = ['leg_right_1_joint', 'leg_right_2_joint', 'leg_right_3_joint', 'leg_right_4_joint',
                            'leg_right_5_joint', 'leg_right_6_joint']
    else:
        joints_desired_l = ['LHipSag', 'LHipLat', 'LHipYaw', 'LKneeSag', 'LAnkLat', 'LAnkSag']
        joints_desired_r = ['RHipSag', 'RHipLat', 'RHipYaw', 'RKneeSag', 'RAnkLat', 'RAnkSag']

    idr = []
    idl = []
    for i in range(0, len(joints_desired_r)):
        idl.append(robot.model.getJointId(joints_desired_l[i]))
        idr.append(robot.model.getJointId(joints_desired_r[i]))
    # print("left leg joint id in pinocchio", idl)
    # print("right leg joint id in pinocchio", idr)

    ##### reset base pos and orn,  only workable in floating-based model
    des_base_ori = [0,0, math.pi]
    base_arr = RotMatrixfromEulerx(des_base_ori)
    print(base_arr)
    robot.model.jointPlacements[1] = pin.SE3(base_arr, np.array([1, 0, 0.8]))

    # robot.model.jointPlacements[1] = pin.SE3(np.eye(3), np.array([1, 0, 0.8]))

    ##### right, lef leg end joints
    q = robot.q0
    pin.forwardKinematics(robot.model, robot.data, q)
    pr = robot.data.oMi[idr[5]].translation
    pl = robot.data.oMi[idl[5]].translation
    print("right ankle joint pos:", pr)
    print("left ankle joint pos:", pl)
    print(" base_position:", robot.model.jointPlacements[1])
    ############################### pinocchio load finish !!!!!!!!!!!!!!!!!!!!!!!!!!! #####################################################
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!pinocchio load urdf done!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #################################################################################################################

    demo_filename = mesh_dirx + '/referdata_swing_mod.txt'
    nlp_traj_filename = mesh_dirx + '/robot_nlp_traj.txt'
    kmp_traj_filename = mesh_dirx + '/robot_kmp_traj.txt'
    traj_filename = mesh_dirx + '/robot_traj.txt'
    angle_filename = mesh_dirx + '/robot_joint_angle.txt'
    state_filename = mesh_dirx + '/robot_state.txt'
    ###################################################-----------------------------###########################
    ################################################### pybullet simulation loading ###########################
    ### intial pose for talos in pybullet
    ###  by default, stepsimulation() is used
    sim_rate = 200
    dt = 1. / sim_rate
    sim_env = SimEnv(sim_rate=sim_rate)

    if (robotname == 'Talos'):
        urobtx = SimRobot(urdfFileName=urdf_filename,
                          basePosition=[0, 0, 1.08],
                          baseRPY=[0, 0, 0])
    else:
        urobtx = SimRobot(urdfFileName=urdf_filename,
                          basePosition=[0, 0, 0.55],
                          baseRPY=[0, 0, 0])

    robotidx = urobtx.id

    num_joint = urobtx.getNumActuatedJoints()
    Homing_pose = np.zeros(num_joint)
    if Full_body_simu:
        if (robotname == 'Talos'):
            ### upper arm
            Homing_pose[4] = 0.2
            Homing_pose[5] = 0.1
            Homing_pose[6] = -0.7
            Homing_pose[7] = -0.6
            Homing_pose[11] = -0.2
            Homing_pose[12] = -0.1
            Homing_pose[13] = 0.7
            Homing_pose[14] = -0.6
            ## lower leg
            Homing_pose[-10] = -0.17644
            Homing_pose[-9] = 0.36913
            Homing_pose[-8] = -0.18599
            Homing_pose[-4] = -0.17644
            Homing_pose[-3] = 0.36913
            Homing_pose[-2] = -0.18599
        else:
            if(Forearm):
                ### upper arm: for whole-body coman
                Homing_pose[3] = -0.2
                Homing_pose[4] = -0.1
                Homing_pose[5] = 0
                Homing_pose[6] = -0.6
                Homing_pose[10] = -0.2
                Homing_pose[11] = 0.1
                Homing_pose[12] = 0
                Homing_pose[13] = -0.6
                ## lower leg
                Homing_pose[-12] = -0.48444
                Homing_pose[-9] = 0.96913
                Homing_pose[-7] = -0.48599
                Homing_pose[-6] = -0.48444
                Homing_pose[-3] = 0.96913
                Homing_pose[-1] = -0.48599
            else:
                ### upper arm: for coman without forearm
                Homing_pose[3] = -0.2
                Homing_pose[4] = -0.1
                Homing_pose[5] = 0
                Homing_pose[6] = -0.6
                Homing_pose[7] = -0.2
                Homing_pose[8] = 0.1
                Homing_pose[9] = 0
                Homing_pose[10] = -0.6                
                ## lower leg
                Homing_pose[-12] = -0.48444
                Homing_pose[-9] = 0.96913
                Homing_pose[-7] = -0.48599
                Homing_pose[-6] = -0.48444
                Homing_pose[-3] = 0.96913
                Homing_pose[-1] = -0.48599

    else:
        if (robotname == 'Talos'):
            ## lower leg
            Homing_pose[2] = -0.2
            Homing_pose[3] = 0.4
            Homing_pose[4] = -0.2
            Homing_pose[8] = -0.2
            Homing_pose[9] = 0.4
            Homing_pose[10] = -0.2
        else:
            ## lower leg
            Homing_pose[2] = -0.2
            Homing_pose[3] = 0.4
            Homing_pose[4] = -0.2
            Homing_pose[8] = -0.2
            Homing_pose[9] = 0.4
            Homing_pose[10] = -0.2

    print("Homing_pose:", Homing_pose)
    t_homing = 2

    useRealTimeSimulation = 0
    pybullet.setRealTimeSimulation(useRealTimeSimulation)

    ############################## enable ankle pressure sensory
    full_joint_number = urobtx.getNumJoints()
    rig_legid = full_joint_number - 2
    left_legid = full_joint_number - 9
    pybullet.enableJointForceTorqueSensor(bodyUniqueId=robotidx, jointIndex=rig_legid, enableSensor=1)
    pybullet.enableJointForceTorqueSensor(bodyUniqueId=robotidx, jointIndex=left_legid, enableSensor=1)

    #####################################################################################################33
    print("!!!!!!!!!!!!!!!!!pybullet load environment done!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #####################################################################################################33

    ##### Gait_Controller
    Controller_ver = Gait_Controller(urbodx=urobtx, id=robotidx, robotnamex=robotname, verbose=True, )

    trailDuration = 10
    prevPose = [0, 0, 0]
    prevPose1 = [0, 0, 1.08]
    hasPrevPose = 0

    t = 0.
    i = 0
    pr_homing_fix = []
    pl_homing_fix = []
    base_home_fix = []

    torso1_linkid = 0
    left_sole_linkid = 36
    right_sole_linkid = 42
    #####################################################################################################33
    print("!!!!!!!!!!!!!!!!!!!!!!!Controller setup done!!!!!!!!!!!!!!!!!!!!!!!!!!")
    ############################ step parameters -----------------#################################################33
    nstep = 20
    dt_sample = dt
    dt_nlp = 0.025
    hcomx = 0.46 #0.5245
    sx = 0.1
    sy = 0.1452
    sz = 0
    st = 0.599
    lift_height = 0.06
    falling_flag = 0


    ### KMP initiallization
    rleg_traj_refx = np.loadtxt(demo_filename)  ### linux/ubuntu
    lleg_traj_refx = np.loadtxt(demo_filename)  ### linux/ubuntu

    # rleg_traj_refx = np.loadtxt(r'D:\research\numerical optimization_imitation learning\nlp_nao_experiments\referdata_swing_mod.txt') ###window
    # lleg_traj_refx = np.loadtxt(r'D:\research\numerical optimization_imitation learning\nlp_nao_experiments\referdata_swing_mod.txt')  ###window

    inDim = 1  ### time as input
    outDim = 6  ### decided by traj_Dim * (pos+?vel+?acc: indicated by pvFlag)
    kh = 2
    lamda = 1
    pvFlag = 1  ## pvFlag = 0(pos) & pvFlag = 1 (pos+vel)


    com_fo_nlp =  NLP(nstep,dt_sample,dt_nlp,hcomx, sx,sy,sz,st,rleg_traj_refx, lleg_traj_refx, inDim, outDim, kh, lamda, pvFlag)
    print(com_fo_nlp.Nsum1)
    outx = np.zeros([com_fo_nlp.Nsum, 12])
    RLfoot_com_pos = np.zeros([9, com_fo_nlp.Nsum1])
    t_long = np.arange(0,com_fo_nlp.Nsum1*dt,dt)

    t_n = int(2 * np.round(com_fo_nlp.Ts[0, 0] / dt_sample))
    #####################################################################################################33
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NLP-KMP-CLASS setup done!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #####################################################################################################33

    FileLength = com_fo_nlp.Nsum1

    traj_opt = np.zeros([FileLength, 12])  ### trajectory reference generated by gait planner
    joint_opt = np.zeros([FileLength, num_joint])  #### joint angle by IK
    state_feedback = np.zeros([FileLength, 40])  ### robot state estimation
    links_pos_prev = np.zeros([full_joint_number + 1, 3])  ### links com position; the last one is the base position
    links_vel_prev = np.zeros([full_joint_number + 1, 3])  ### links com velocities
    support_flag = np.zeros([FileLength, 1])
    gcom_pre = [0, 0, 0]
    com_ref_base = [0, 0, 0]
    com_feedback_base = [0, 0, 0]
    com_ref_det = [0, 0, 0]
    com_ref_det_pre = [0, 0, 0]
    com_feedback_det = [0, 0, 0]
    com_feedback_det_pre = [0, 0, 0]

    angle_ref_det = [0, 0, 0]
    angle_feedback_det = [0, 0, 0]
    angle_ref_det_pre = [0, 0, 0]
    angle_feedback_det_pre = [0, 0, 0]



    sim_env.resetCamera() 
    ############################################################################################# main loop for robot gait generation and control ##########################################
    while i < FileLength:
        if (useRealTimeSimulation):
            # dt = datetime.now()
            # t = (dt.second / 60.) * 2. * math.pi
            t = t + dt
        else:
            t = t + dt

        #################===============================================================================
        ############## robot control loop##################################################
        if t < t_homing:  ############# initial pose
            Homing_pose_t = Homing_pose * math.sin(t / t_homing / 2. * math.pi)
            q = robot.q0
            if Freebase:  ### note the in pinocchio, freebase has seven DoF
                if Full_body_simu:
                    q[0 + 7:6 + 7] = Homing_pose_t[-12:-6]
                    q[6 + 7:12 + 7] = Homing_pose_t[-6:]
                else:
                    q[0 + 7:12 + 7] = Homing_pose_t
            else:
                # q = robot.q0
                if Full_body_simu:
                    q[0:6] = Homing_pose_t[-12:-6]
                    q[6:12] = Homing_pose_t[-6:]
                else:
                    q = Homing_pose_t * math.sin(t / t_homing / 2. * math.pi)

            joint_opt[i] = Homing_pose_t
            urobtx.setActuatedJointPositions(Homing_pose_t)
            base_pos_m = pybullet.getBasePositionAndOrientation(robotidx)[0]
            # left_sole_urdf_position = p.getLinkState(bodyUniqueId=robotidx, linkIndex=6)[4]
            # right_sole_urdf_position = p.getLinkState(bodyUniqueId=robotidx, linkIndex=13)[4]
            # base_homing = np.array(lsx_ori)
            if Freebase:
                robot.model.jointPlacements[1] = pin.SE3(np.eye(3), np.array(base_pos_m))
            pin.forwardKinematics(robot.model, robot.data, q)
            pr_homing = robot.data.oMi[idr[5]].translation
            pl_homing = robot.data.oMi[idl[5]].translation
            base_home_fix = base_pos_m
            pl_homing_fix = tuple(pl_homing)
            pr_homing_fix = tuple(pr_homing)
            # print('right_leg_sole_position:', pl_homing_fix)
            # print('left_leg_sole_position:', pl_homing_fix)


            links_pos, links_vel, links_acc = Controller_ver.get_link_vel_vol(i, dt, links_pos_prev, links_vel_prev)
            links_pos_prev = links_pos
            links_vel_prev = links_vel

            support_flag[i] = 0  ### 0, double support, right support

            ###################state feedbacke ###########################################
            gcom_m, right_sole_pos, left_sole_pos, base_pos_m, base_angle_m, right_ankle_force, left_ankle_force, gcop_m, support_flag, dcm_pos_m, com_vel_m, links_pos, links_vel, links_acc = \
                Controller_ver.state_estimation(i, dt, support_flag, links_pos_prev, links_vel_prev, gcom_pre)

            links_pos_prev = links_pos
            links_vel_prev = links_vel
            gcom_pre = gcom_m
            com_feedback_base = gcom_m
            com_ref_base = base_pos_m
        else:
            ###################state feedbacke ###########################################
            gcom_m, right_sole_pos, left_sole_pos, base_pos_m, base_angle_m, right_ankle_force, left_ankle_force, gcop_m, support_flag, dcm_pos_m, com_vel_m, links_pos, links_vel, links_acc = \
                Controller_ver.state_estimation(i, dt, support_flag, links_pos_prev, links_vel_prev, gcom_pre)



            links_pos_prev = links_pos
            links_vel_prev = links_vel
            gcom_pre = gcom_m




            if (((abs(base_angle_m[0]) >= 30 * math.pi / 180) or (
                    abs(base_angle_m[1]) >= 30 * math.pi / 180)) and (falling_flag<0.5)):  ### falling down
                falling_flag  = 1
                np.savetxt(traj_filename, traj_opt, fmt='%s', newline='\n')
                np.savetxt(angle_filename, joint_opt, fmt='%s', newline='\n')
                np.savetxt(state_filename, state_feedback, fmt='%s', newline='\n')
                np.savetxt(nlp_traj_filename, outx, fmt='%s', newline='\n')
                np.savetxt(kmp_traj_filename, RLfoot_com_pos, fmt='%s', newline='\n')
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Robot is falling, save data!!!!!!!!!!!!!!!!!!!!!!!!11")


            ######## reference trajectory generation #############################
            if Freebase: #routine1: change the base position and orientation for pinocchio IK: time-cost process due to the redundant freedom
                ################# test
                # des_base = np.array([0.02 * (math.sin((t - t_homing) * 400 * math.pi / 180)),
                #                    0.02 * (math.sin((t - t_homing) * 400 * math.pi / 180)),
                #                    0.02 * abs(math.sin((t - t_homing) * 400 * math.pi / 180))]) + np.array(base_home_fix)
                # robot.model.jointPlacements[1] = pin.SE3(np.eye(3), des_base)
                des_pl = np.array(pl_homing_fix)
                oMdesl = pin.SE3(np.eye(3), des_pl)
                des_pr = np.array(pr_homing_fix)
                oMdesr = pin.SE3(np.eye(3), des_pr)
                if (t-t_homing<=0.01):
                    print("right_foot_pose_fix:", pr_homing_fix)
                    print("left_foot_pose_fix:",pl_homing_fix)
                    print("base_home_fix:", base_home_fix)
                

                j = i + 1- (int)(round(t_homing/dt)) #### current j-th step in the low-level control
                state_feedback[j, 0:3] = gcom_m
                state_feedback[j, 3:6] = right_sole_pos
                state_feedback[j, 6:9] = left_sole_pos
                state_feedback[j, 9:15] = right_ankle_force
                state_feedback[j, 15:21] = left_ankle_force
                state_feedback[j, 21:24] = gcop_m
                state_feedback[j, 24:27] = dcm_pos_m
                state_feedback[j, 27:30] = com_vel_m
                state_feedback[j, 30:33] = base_pos_m
                state_feedback[j, 33:36] = base_angle_m
                # print('base_link_position:',base_pos_m)
                # print('right_leg_sole_position:', right_sole_pos)
                # print('left_leg_sole_position:', left_sole_pos)


                #############---------------------# MPC gait --------------------------########################
                # cpptest = "/home/jiatao/Dropbox/nmpc_pybullet/build/src/MPC_WALK.exe"  # in linux without suffix .exe
                # donser = run_nmpc_external_ext(j,cpptest)
                #
                # if len(donser) == 12:
                #     traj_opt[i] = donser
                # else:
                #     donser = traj_opt[i-1]
                # des_base = donser[0:3]
                # des_base_ori = donser[3:6]/10
                # xxx = Controller_ver.RotMatrixfromEuler(des_base_ori)
                # # robot.model.jointPlacements[1] = pin.SE3(xxx, des_base)
                # robot.model.jointPlacements[1] = pin.SE3(np.eye(3), des_base)
                # des_pr = donser[6:9]
                # des_pl = donser[9:12]
                #
                # if (i<=1):
                #     com_ref_base = des_base
                #     com_feedback_base = gcom_m
                # # des_base_ori = donser[3:6]/10
                # # xxx = Controller_ver.RotMatrixfromEuler(des_base_ori)
                # # robot.model.jointPlacements[1] = pin.SE3(xxx, des_base)                
                
                ##----------------Mosek NLP gait generation --------------------###################
                j_index = int(np.floor(
                    (j) / (com_fo_nlp.dt / com_fo_nlp.dtx)))  ####walking time fall into a specific optmization loop
                if ((j_index >= 1) and (abs(j * com_fo_nlp.dtx - j_index * com_fo_nlp.dt) <= 0.8 * dt_sample)):
                    res_outx = com_fo_nlp.nlp_nao(j_index)
                    outx[j_index - 1, :] = res_outx


                # ##------ kMP based swing leg trajectory -----######### 
                # rfoot_p, lfoot_p = com_fo_nlp.kmp_foot_trajectory(j, dt_sample, j_index, rleg_traj_refx, lleg_traj_refx,
                #                                                   inDim, outDim, kh, lamda, pvFlag,lift_height)

                ##------ kMP based swing leg trajectory -----######### 
                rfoot_p, lfoot_p = com_fo_nlp.Bezier_foot_trajectory(j, dt_sample, j_index, lift_height)

                RLfoot_com_pos[0:3, j - 1] = copy.deepcopy(rfoot_p[0:3, 0])
                RLfoot_com_pos[3:6, j - 1] = copy.deepcopy(lfoot_p[0:3, 0])

                com_intex = com_fo_nlp.XGetSolution_CoM_position(j, dt_sample, j_index)
                RLfoot_com_pos[6:9, j - 1] = copy.deepcopy(com_intex[0:3, 0])

                #### modified Catesian position for body and leg        
                des_base = copy.deepcopy(com_intex[0:3, 0])
                des_base[0] = copy.deepcopy((-com_intex[0, 0]- 0 + base_home_fix[0]))
                des_base[1] = copy.deepcopy((com_intex[1, 0]- 0 + base_home_fix[1]))
                des_base[2] = copy.deepcopy((com_intex[2, 0]- hcomx + base_home_fix[2]))

                robot.model.jointPlacements[1] = pin.SE3(np.eye(3), des_base)
                des_pr = copy.deepcopy(rfoot_p[0:3, 0])
                des_pr[0] = copy.deepcopy((-rfoot_p[0, 0]- 0 + pr_homing_fix[0]))
                des_pr[1] = copy.deepcopy((rfoot_p[1, 0]- (- (sy)*0.5) + pr_homing_fix[1]))
                des_pr[2] = copy.deepcopy((lfoot_p[2, 0]- 0 + pr_homing_fix[2]))
                des_pl = copy.deepcopy(lfoot_p[0:3, 0])
                des_pl[0] = copy.deepcopy((-lfoot_p[0, 0]- 0 + pl_homing_fix[0]))
                des_pl[1] = copy.deepcopy((lfoot_p[1, 0]- ((sy)*0.5) + pl_homing_fix[1]))
                des_pl[2] = copy.deepcopy((rfoot_p[2, 0]- 0 + pl_homing_fix[2]))

                if (i<=1):
                    com_ref_base = des_base
                    com_feedback_base = base_pos_m
            else:  ####routine2: set the based position in local framework(note that always zeros), transform the base function in the
                des_pl = np.array([0.03 * abs(math.sin((t - t_homing) * 5 * math.pi / 180)),
                                   0 - 0.05 * (math.sin((t - t_homing) * 5 * math.pi / 180)),
                                   0.05 * abs(math.sin((t - t_homing) * 5 * math.pi / 180))]) + np.array(pl_homing_fix)
                des_pr = np.array([0.03 * abs(math.sin((t - t_homing) * 5 * math.pi / 180)),
                                   0 - 0.05 * (math.sin((t - t_homing) * 5 * math.pi / 180)),
                                   0.05 * abs(math.sin((t - t_homing) * 5 * math.pi / 180))]) + np.array(pr_homing_fix)
                
                state_feedback[j, 0:3] = gcom_m
                state_feedback[j, 3:6] = right_sole_pos
                state_feedback[j, 6:9] = left_sole_pos
                state_feedback[j, 9:15] = right_ankle_force
                state_feedback[j, 15:21] = left_ankle_force
                state_feedback[j, 21:24] = gcop_m
                state_feedback[j, 24:27] = dcm_pos_m
                state_feedback[j, 27:30] = com_vel_m
                state_feedback[j, 30:33] = base_pos_m
                state_feedback[j, 33:36] = base_angle_m
                # print('base_link_position:',base_pos_m)
                # print('right_leg_sole_position:', right_sole_pos)
                # print('left_leg_sole_position:', left_sole_pos)   
                #                              
            # ################# IK-based feeback control: we can use admittance control#################################33
            # com_ref_det = np.array(des_base) - np.array(com_ref_base)
            # com_feedback_det = np.array(base_pos_m) - np.array(com_feedback_base)
            # angle_ref_det = des_base_ori
            # angle_feedback_det = base_angle_m
            
            # det_comxxxx, det_body_anglexxxx =Controller_ver.CoM_Body_pd(dt,com_ref_det, com_feedback_det, com_ref_det_pre, com_feedback_det_pre, angle_ref_det,angle_feedback_det, angle_ref_det_pre, angle_feedback_det_pre)
            # des_com_pos_control = det_comxxxx + np.array(des_base)
            # det_base_angle_control = det_body_anglexxxx
            # det_base_matrix_control = Controller_ver.RotMatrixfromEuler(det_base_angle_control)
            # robot.model.jointPlacements[1] = pin.SE3(det_base_matrix_control, des_com_pos_control)
            # # robot.model.jointPlacements[1] = pin.SE3(np.eye(3), des_com_pos_control)
            
            # com_ref_det_pre = com_ref_det
            # com_feedback_det_pre = com_feedback_det
            # angle_ref_det_pre = angle_ref_det
            # angle_feedback_det_pre = angle_feedback_det



            ############ IK-solution for the float-based humanod: providing initial guess "homing_pose" ------#################
            ########### set endeffector id for ik using pinocchio
            JOINT_IDl = idl[5]
            JOINT_IDr = idr[5]
            oMdesl = pin.SE3(np.eye(3), des_pl)
            oMdesr = pin.SE3(np.eye(3), des_pr)

            des_pr_l = copy.deepcopy(des_pl)
            des_pr_l[1] =  copy.deepcopy(des_pl[1] - sy)
            oMdesr_l = pin.SE3(np.eye(3), des_pr_l)
            if (robotname == 'Talos'):
                q_ik = joint_lower_leg_ik(robot, oMdesl, JOINT_IDl, oMdesr, JOINT_IDr, Freebase, Homing_pose,t,t_homing,robotname)
            else:
                q_ik = joint_lower_leg_ik(robot, oMdesr_l, JOINT_IDr, oMdesr, JOINT_IDr, Freebase, Homing_pose, t,
                                          t_homing,robotname)
            # print('q_ik:',q_ik)
            ######## joint command: position control mode ###########################
            joint_opt[i] = q_ik
            urobtx.setActuatedJointPositions(q_ik)

            ###########################===========================================================
            #########################################################################################

        i += 1

        ########## plot ########################################
        if (i == FileLength - 1):
            np.savetxt(traj_filename, traj_opt, fmt='%s', newline='\n')
            np.savetxt(angle_filename, joint_opt, fmt='%s', newline='\n')
            np.savetxt(state_filename, state_feedback, fmt='%s', newline='\n')
            np.savetxt(nlp_traj_filename, outx, fmt='%s', newline='\n')
            np.savetxt(kmp_traj_filename, RLfoot_com_pos, fmt='%s', newline='\n')
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Process in ending, save data!!!!!!!!!!!!!!!!!!!!!!!!11")

            #-------------------------# CoM plots #-------------------------__#

            fig,axs2 = plt.subplots(3,constrained_layout=True,figsize=(6.5,7))
            plt.rcParams.update({'font.size': 14})
            axs2[0].plot(t_long,RLfoot_com_pos[0,:],'g',label="rfootx")
            axs2[0].plot(t_long,RLfoot_com_pos[3,:],'r',label="lfootx")
            axs2[0].plot(t_long,RLfoot_com_pos[6,:],'b',label="comx")
            axs2[0].plot(t_long,state_feedback[:,30],label="pelvisx")
            
            axs2[0].set_xlabel('Time (s)', fontsize=12)
            axs2[0].legend(loc='upper left')
            axs2[0].grid()

            axs2[1].plot(t_long,RLfoot_com_pos[1,:],'g',label="rfooty")
            axs2[1].plot(t_long,RLfoot_com_pos[4,:],'r',label="lfooty")
            axs2[1].plot(t_long,RLfoot_com_pos[7,:],'b',label="comy")
            axs2[1].plot(t_long,state_feedback[:,31],label="pelvisy")
            axs2[1].set_xlabel('Time (s)', fontsize=12)
            axs2[1].legend(loc='upper left')
            axs2[1].grid()

            axs2[2].plot(t_long,RLfoot_com_pos[2,:],'g',label="rfootz")
            axs2[2].plot(t_long,RLfoot_com_pos[5,:],'r',label="lfootz")
            axs2[2].plot(t_long,RLfoot_com_pos[8,:],'b',label="comz")
            axs2[2].plot(t_long,state_feedback[:,32],label="pelvisz")
            axs2[2].set_xlabel('Time (s)', fontsize=12)
            axs2[2].legend(loc='upper left')
            axs2[2].grid()
            # if SAVE_DATA:
            #     plt.savefig(fname + '/orientation_Trial_' + str(trial_num) + ".pdf",dpi=600, bbox_inches = "tight")

            plt.show()



        ##### doesn't use it in realtime simu mode
        pybullet.stepSimulation()

        ##############only work in real-time model???? a bug #########################3
        # ls = pybullet.getLinkState(robotidx, torso1_linkid)
        # print("torso_link:",ls[0])
        # if (hasPrevPose):
        #     # pybullet.addUserDebugLine(prevPose, lsx, [0, 0, 0.3], 1, trailDuration)
        #     pybullet.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
        # # prevPose = lsx_ori
        # prevPose1 = ls[4]
        # hasPrevPose = 1




if __name__ == "__main__":
    main()
