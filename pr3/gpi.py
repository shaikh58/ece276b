import numpy as np
from main import car_next_state
from utils import visualize
from time import time
import matplotlib.pyplot as plt
import itertools
import copy

GRID_X, GRID_Y, GRID_TH = 30, 30, 30
T = 100
GAMMA = 1
ref_traj = np.load('./ref_traj.npy')
time_step = 0.5
x_car,y_car,theta_car = 1.5,0.0,np.pi/2
arr_car_state = np.zeros((240,3))
arr_car_state[0] = np.array([x_car,y_car,theta_car])
POL = np.ones((T,GRID_X, GRID_Y, GRID_TH,2))*0.1

x_grid, y_grid = np.linspace(-3,3,30), np.linspace(-3,3,30)
th_grid = np.linspace(-np.pi,np.pi,30)
t,x,y,th = np.meshgrid(np.arange(100),x_grid,y_grid,th_grid)
grid = np.vstack([t.ravel(), x.ravel(), y.ravel(),th.ravel()])
u_v, u_w = np.meshgrid(np.linspace(0,1,10),np.linspace(-1,1,20))
controls = np.vstack([u_v.ravel(), u_w.ravel()])

def stage_cost(states,ctrls,policy,follow_policy=False):
    '''vectorized stage cost calculation for policy improvement step i.e. l(x,u) for all x,u'''
    pos_error_terms = np.linalg.norm(states[1:3,:].T, axis=1) ** 2
    angle_error_terms = (1-np.cos(states[3]))**2
    error_terms = pos_error_terms + angle_error_terms
    if follow_policy:
        ctrl_terms = np.linalg.norm(policy,axis=4).flatten()
        total_stage_cost = error_terms + ctrl_terms
    else:
        ctrl_terms = np.linalg.norm(ctrls.T, axis=1) ** 2
        repeated_error_terms = np.broadcast_to(error_terms,
                                          (ctrl_terms.shape[0], error_terms.shape[0]))
        repeated_ctrl_terms = np.broadcast_to(ctrl_terms, (pos_error_terms.shape[0], ctrl_terms.shape[0]))
        total_stage_cost = repeated_error_terms + repeated_ctrl_terms.T

    return total_stage_cost

# def motion_model(states, ctrl, follow_policy=False):
#     cos_term = np.cos(grid[3] + np.repeat(ref_traj[0:100, 2], 27000)) * time_step
#     sin_term = np.sin(grid[3] + np.repeat(ref_traj[0:100, 2], 27000)) * time_step
#     repeated_cos_term = np.broadcast_to(cos_term, (controls.shape[1], cos_term.shape[0]))
#     repeated_sin_term = np.broadcast_to(sin_term, (controls.shape[1], sin_term.shape[0]))
#     gu_vx = repeated_cos_term.T * controls[0]
#     gu_vy = repeated_sin_term.T * controls[0]
#     gu_w = time_step * np.repeat(controls[1], 2700000 / controls[1].shape[0])
#     gu_w_bdcst = np.broadcast_to(gu_w, (controls[1].shape[0], gu_w.shape[0]))
#     gu = np.stack([gu_vx.T, gu_vy.T, gu_w_bdcst], axis=1)
#     bcst_state = np.broadcast_to(grid[1:4, :], (200, 3, 2700000))
#     bcst_state + gu +
#     return
def get_motion_model():
    list_u = list(itertools.product(np.linspace(0,1,10),np.linspace(-1,1,20)))
    arr_next_states = np.zeros((len(list_u),T*GRID_X*GRID_Y*GRID_TH,4))
    a = 0
    for u in list_u:
        s = 0
        for t in range(T):
            for x in x_grid:
                for y in y_grid:
                    for angle in th_grid:
                        if s%100000==0:
                            print(a,s)
                        err_x = x + time_step*np.cos(angle + ref_traj[t,2])*u[0] + (ref_traj[t,0] - ref_traj[t+1,0])
                        err_y = y + time_step*np.sin(angle + ref_traj[t,2])*u[0] + (ref_traj[t,1] - ref_traj[t+1,1])
                        err_angle = angle + time_step*u[1] + (ref_traj[t,2] - ref_traj[t+1,2])
                        wrapped_angle = (err_angle + np.pi) % (2 * np.pi) - np.pi
                        arr_next_states[a,s] = (t+1,err_x,err_y,wrapped_angle)
                        s += 1
        a += 1
    return list_u, arr_next_states


# create mapping from states to index
map_state_ix = {}
map_ix_state = {}
c=0
for t in range(T):
    print(t)
    for x in x_grid:
        for y in y_grid:
            for angle in th_grid:
                map_state_ix[(t,x,y,np.round(angle,5))] = c
                map_ix_state[c] = (t,x,y,np.round(angle,5))
                c+=1
#
arr_next_states = get_motion_model()
# # arr_next_states = np.load('./motion_model.npy')
# # # make sure each next state from motion model is a valid state
for u in range(200):
    print(u)
    x_diff = np.broadcast_to(x_grid,(2700000,30)).T - np.broadcast_to(arr_next_states[u][:, 1],(30,2700000))
    all_x_prime_ix = np.argmin(np.abs(x_diff), axis=0)
    all_x_prime = x_grid[all_x_prime_ix]

    y_diff = np.broadcast_to(y_grid, (2700000, 30)).T - np.broadcast_to(arr_next_states[u][:, 2], (30, 2700000))
    all_y_prime_ix = np.argmin(np.abs(y_diff), axis=0)
    all_y_prime = y_grid[all_y_prime_ix]

    th_diff = np.broadcast_to(th_grid, (2700000, 30)).T - np.broadcast_to(arr_next_states[u][:, 3], (30, 2700000))
    all_th_prime_ix = np.argmin(np.abs(th_diff), axis=0)
    all_th_prime = th_grid[all_th_prime_ix]

    arr_next_states[u] = np.stack([arr_next_states[u][:, 0], all_x_prime, all_y_prime, all_th_prime]).T
# # np.save('./arr_next_states_valid.npy',arr_next_states)
#
# # convert the 4-tuple state to an index using the mapping, so that arr_next_states_valid is 2.7M x 1 vector
# arr_next_states_valid = np.load('./arr_next_states_valid.npy')
arr_next_states_valid = copy.deepcopy(arr_next_states)
arr_next_states_ix = np.zeros((200,2700000))
for u in range(200):
    print(u)
    for i in range(2700000):
        arr_next_states_ix[u,i] = map_state_ix[tuple((int(arr_next_states_valid[u,i][0])%100,arr_next_states_valid[u,i][1],
                                                     arr_next_states_valid[u,i][2],np.round(arr_next_states_valid[u, i][3],5)))]
# np.save('./arr_next_states_ix.npy',arr_next_states_ix)
# stage_cost = stage_cost(grid,controls,POL)
# np.save('./stage_cost.npy',stage_cost)

# stage_cost = np.load('./stage_cost.npy')
# arr_next_states_ix = np.load('./arr_next_states_ix_int.npy')
V = np.zeros((T*GRID_X*GRID_Y*GRID_TH))
# # value iteration
num_iter = 100
for v in range(num_iter):
    start = time()
    Q = stage_cost + V[arr_next_states_ix]
    V_next = np.min(Q, axis=0)
    print(np.linalg.norm(V_next - V))
    if v==num_iter-1:
        optimal_policy = np.argmin(Q, axis=0)
        break
    V = V_next
    time_taken = time() - start
    print('Iteration {x}: {y}s'.format(x=str(v),y=str(time_taken)))

# np.save('./value_fcn_{x}_iter.npy'.format(x=str(num_iter)), V)
# np.save('./optimal_policy_{x}_iter.npy'.format(x=str(num_iter)), optimal_policy)

list_u = list(itertools.product(np.linspace(0,1,10),np.linspace(-1,1,20)))
ix_ctrl_map = {}
c=0
for u in list_u:
    ix_ctrl_map[c] = u
    c+=1
opt_pol_val = np.zeros((2700000,2))
for i in range(2700000):
    opt_pol_val[i] = ix_ctrl_map[optimal_policy[i]]
#
arr_car_state = np.zeros((240,4))
arr_car_state[0] = np.array([0,x_car,y_car,theta_car])
for t in range(239):
    # first snap the state to a valid grid position
    arr_car_state[t, 0] = arr_car_state[t, 0] % 100
    arr_car_state[t, 1] = x_grid[np.argmin(np.abs(x_grid - arr_car_state[t,1]))]
    arr_car_state[t, 2] = y_grid[np.argmin(np.abs(y_grid - arr_car_state[t, 2]))]
    arr_car_state[t, 3] = th_grid[np.argmin(np.abs(th_grid - ((arr_car_state[t, 3] + np.pi) % (2 * np.pi) - np.pi)))]
    if arr_car_state[t, 0]==0:
        # since map_state_ix starts from t=1 and ends at t=100, t=100 is actually t=0 so can be interchanged
        state_ix = map_state_ix[(0,arr_car_state[t,1],arr_car_state[t,2],np.round(arr_car_state[t,3],5))]
    else:
        state_ix = map_state_ix[(int(arr_car_state[t,0]),arr_car_state[t,1],arr_car_state[t,2],
                                 np.round(arr_car_state[t,3],5))]
    arr_car_state[t+1,0] = t + 1
    arr_car_state[t+1,1:] = car_next_state(time_step, arr_car_state[t,1:], opt_pol_val[state_ix], noise=False)
#
# # visualize trajectory based on optimal policy
obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
visualize(arr_car_state[:,1:], ref_traj, obstacles, np.zeros((240,)), time_step, save=True)

# get trajectory error
arr_car_state[:,2] = arr_car_state[:,2]%(2*np.pi)
err = np.linalg.norm(ref_traj[:-1] - arr_car_state[:-1], axis=1).sum()

x = [12820,11981,11323,10731,10154,9584,9057,8609,8258,7996,7792,7637,7514,7410,7335,7283,7245,7220,7201,7183]

plt.plot(x)
plt.xlabel('Iteration #')
plt.ylabel('L2 norm of V_i - V_i-1')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])

12820.220740672748
Iteration 0: 186.55369925498962s
11981.183893831663
Iteration 1: 242.25970721244812s
11323.054228134557
Iteration 2: 198.3515260219574s
10731.721780451866
Iteration 3: 249.68393516540527s
10154.316761600636
Iteration 4: 276.16519498825073s
9584.14951817629
Iteration 5: 385.149178981781s
9057.710739995862
Iteration 6: 347.18753004074097s
8609.768756513096
Iteration 7: 464.4534990787506s
8258.97755378909
Iteration 8: 476.82827711105347s
7996.041673017711
Iteration 9: 226.2155110836029s
7792.342453824348
Iteration 10: 364.96212887763977s
7637.582258097961
Iteration 11: 239.90075206756592s
7514.759332260042
Iteration 12: 331.4890847206116s
7410.676935391962
Iteration 13: 316.36711025238037s
7335.851617025516
Iteration 14: 318.1770889759064s
7283.255884633438
Iteration 15: 378.2142219543457s
7245.151545290966
Iteration 16: 245.7545521259308s
7220.182680192333
Iteration 17: 237.80629301071167s
7201.322851049962
Iteration 18: 378.0293798446655s
7183.747182527407
Iteration 19: 246.64441180229187s