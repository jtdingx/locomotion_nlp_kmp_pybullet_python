import cvxpy as cp
import numpy as np
from cvxpygen import cpg
# from cvxpygen_master.cvxpygen import cpg
import pinocchio as pin


def generate_optimal_controller():

    n = 12 #state size:  x,y,z,roll,pitch,yaw,x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot
    m = 6 ## control input size

    N = 7  #### future steps
    dt= 0.005

    n_cmp = 2 ### cmp size
    m_inertial = 3 ### initial matrix size


    ###### constraints
    np.random.seed(0)
    


    mass = 48.7
    grav = np.array([0,0,-9.81])
    g = 9.81

    ##### rotation angle constraint
    pitch_max = 15.0/180*np.pi
    pitch_min = -5.0/180*np.pi
    roll_max = 10.0/180*np.pi
    roll_min = -10.0/180*np.pi
    # pitch_max = 15.0/180*np.pi
    # pitch_min = -15.0/180*np.pi
    ### torque constraint
    torque_x_max = 80
    torque_x_min = -50
    torque_y_max = 50
    toruqe_y_min = -50


    X = cp.Variable((n,N+1),name="X") ## x,y,z,roll,pitch,yaw,x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot in the prediction window
    
    X_cmp = cp.Variable((n_cmp,N+1),name="X_cmp")

    U = cp.Variable((m,N),name="U")   ## x_ddot,y_ddot,z_ddot,roll_dot,pitch_dot,yaw_dot
 

    ###### Matrices initialization: reshape, initialization ### 
    ### recieved parameterize from real-time loop ####################
    x_init = cp.Parameter(n,name='x_init')    
    # x_init.value = np.zeros(n) #### current state 
    A_dyn = cp.Parameter((n*N,n),name='A_dyn')
    # A_dyn.value = np.zeros((n*N,n)) #### evolved with time-varying referenece rody rotation state
    X_ref = cp.Parameter((n,N),name='X_ref')
    # X_ref.value = np.zeros((n,N)) ### reference CoM state in the prediction window


    J_ini = cp.Parameter((m_inertial,m_inertial),name='J_ini')
    # J_ini.value = np.zeros((m_inertial,m_inertial))
    L_ref = cp.Parameter((n_cmp,N),name='L_ref')
    # L_ref = np.zeros((n_cmp,N))
    Z_c = cp.Parameter(1,name='Z_c')
    # Z_c = 0.46

    B_dyn = np.array([ [0.5*dt**2,0,0,0,0,0], 
                    [0,0.5*dt**2,0,0,0,0], 
                    [0,0,0.5*dt**2,0,0,0], 
                    [0,0,0,0.5*dt**2,0,0],
                    [0,0,0,0,0.5*dt**2,0],
                    [0,0,0,0,0,0.5*dt**2],
                    [dt,0,0,0,0,0],
                    [0,dt,0,0,0,0],
                    [0,0,dt,0,0,0],
                    [0,0,0,dt,0,0],
                    [0,0,0,0,dt,0],
                    [0,0,0,0,0,dt]])

    constr = []


    constr += [X[:,0] == x_init]
    for i in range(N):
        constr += [X[:,i+1] == A_dyn[i*n:(i+1)*n,:]@X[:,i] + B_dyn@U[:,i]] ### state transition
        constr += [X_cmp[0,i+1] == X[0,i+1] - U[0,i]/g * Z_c - J_ini[1,1]*U[4,i]/(mass*g)] ### cmpx computing
        constr += [X_cmp[1,i+1] == X[1,i+1] - U[1,i]/g * Z_c + J_ini[0,0]*U[3,i]/(mass*g)] ### cmpy computing
        
        constr += [ torque_x_min <= J_ini[1,1]*U[4,i], J_ini[1,1]*U[4,i]<= torque_x_max]
        constr += [ toruqe_y_min <= J_ini[0,0]*U[3,i], J_ini[0,0]*U[3,i]<= torque_y_max]
        constr += [ roll_min <= X[3,i+1], X[3,i+1]<= roll_max]
        constr += [ pitch_min <= X[4,i+1], X[4,i+1]<= pitch_max]
        constr += [X[2,i+1] == X_ref[2,i]] ### no change of height

        #### maybe we can try limit the CMP



    #####-----position mpc
    R = cp.Parameter((m,m),name="R")
    R = .4*np.array([0.7,0.7,0.7,1.0,0.5,1.0])*np.eye(m) #
    Q = 1*np.array([200,200,100,200,200,200,20,20,10,20,20,20])*np.eye(n)
    Q_cmp = 1*np.array([50,50])*np.eye(n_cmp)
    cost = cp.sum_squares(Q@(X[:,1:N+1]-X_ref))+ cp.sum_squares(R@U) + cp.sum_squares(Q_cmp@(X_cmp[:,1:N+1]-L_ref))

    # cost = cp.sum_squares(Q@(X[:,:N]-X_ref))+ cp.sum_squares(R@U) + cp.sum_squares(R_grf@F)

    ### minimize the objective function ########### 
    prob = cp.Problem(cp.Minimize(cost), constr)


    cpg.generate_code(prob,code_dir='mpc_code_gen',solver='OSQP')#,solver='ECOS')


if __name__ == '__main__':
    generate_optimal_controller()