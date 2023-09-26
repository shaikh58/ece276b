from casadi import *
import numpy as np
from main import car_next_state
from utils import visualize
from time import time

Q = np.eye(2)
R = np.eye(2)
q = 1
PLAN_HORIZON = 15
GAMMA = 0.99
ref_traj = np.load('./ref_traj.npy')
time_step = 0.5
x_car,y_car,theta_car = 1.5,0.0,np.pi/2
arr_car_state = np.zeros((240,3))
arr_car_state[0] = np.array([x_car,y_car,theta_car])
opti = casadi.Opti()

# obstacle coordinates
o1 = opti.parameter(2)
o2 = opti.parameter(2)
opti.set_value(o1,np.array([-2,-2]))
opti.set_value(o2,np.array([1,2]))
for t in range(240):
    start = time()
    print('iteration #',t)
    U = opti.variable(2,min(PLAN_HORIZON,240-t))
    E = opti.variable(3,min(PLAN_HORIZON,240-t))
    # reference trajectory
    R = opti.parameter(3, min(PLAN_HORIZON, 240 - t))
    opti.set_value(R, ref_traj.T[:, t:t + min(PLAN_HORIZON, 240 - t)])
    cost_fcn = 0
    # initial error term constraint (moves with every iteration)
    opti.subject_to(E[:,0] == arr_car_state[t]-ref_traj[t])
    # E[:, 0] == arr_car_state[t] - ref_traj[t]
    for i in range(min(PLAN_HORIZON,240-t)-1):
        # constraints on the control input; 0 index is linear vel, 1 index is angular vel
        opti.subject_to([U[0, i] <= 1, U[0, i] >= 0, U[1, i] <= 1, U[1, i] >= -1])
        # obstacle constraints
        opti.subject_to([norm_2(E[0:2,i] + R[0:2,i] - o1) >= 0.5, norm_2(E[0:2,i] + R[0:2,i] - o2) >= 0.5])
        # error motion model constraints
        opti.subject_to(E[0, i + 1] == E[0, i] + time_step * cos(E[2,i]+R[2,i])*U[0,i] + (R[0,i] - R[0,i+1]))
        opti.subject_to(E[1, i + 1] == E[1, i] + time_step * sin(E[2, i] + R[2, i]) * U[0, i] + (R[1, i] - R[1, i + 1]))
        opti.subject_to(E[2, i + 1] == E[2, i] + time_step * U[1,i] + (R[2, i] - R[2, i + 1]))

        cost_fcn += GAMMA **i * (sumsqr(E[0:2, i]) + sumsqr(1 - cos(E[2, i])) + sumsqr(U[:,i]))
    opti.minimize(cost_fcn)
    opti.solver('ipopt',{'ipopt.print_level': 0, 'print_time': 0})
    sol = opti.solve()
    first_u = sol.value(U)[:,0]
    # apply the first control input to the system and get the car's next position
    arr_car_state[t+1] = car_next_state(time_step, arr_car_state[t], first_u, noise=True)

    time_taken = time() - start
    print('Time taken: ', time_taken)

# visualize the trajectory
obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
visualize(arr_car_state, ref_traj, obstacles, np.zeros((240,)), time_step, save=True)

# get tracking error: 785 vs 2180 for simple controller! But took 8 mins to plan vs less than 1 min
arr_car_state[:,2] = arr_car_state[:,2]%(2*np.pi)
err = np.linalg.norm(ref_traj[:-1] - arr_car_state[:-1], axis=1).sum()