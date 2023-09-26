from utils import *
import numpy as np
from copy import deepcopy
import itertools

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

# directions: +ve is down, right i.e. [0,1] = down, [1,0] = right
# 0:right, 1:up, 2:left, 3:down
def partB():
    env_folder = "./environments/random_envs"
    env, info, env_path = load_random_env(env_folder)

env_path = "./environments/known_envs/doorkey-5x5-normal.env"

env, info = load_env(env_path)
# plot_env(env)
door = env.grid.get(info["door_pos"][0], info["door_pos"][1])
door_pos_x, door_pos_y = info["door_pos"][0] + 1, info["door_pos"][1] + 1
key_pos_x, key_pos_y = info["key_pos"][0] + 1, info["key_pos"][1] + 1
is_open = door.is_open
# assume door is always locked in the known_env maps
is_locked = door.is_locked

# DP implementation
num_actions = 5
# actions: MF,TL,TR,PK,UD = 0,1,2,3,4
costs = {0:1, 1:1, 2:1, 3:1, 4:1}
goal_pos = info['goal_pos']
goal_pos_x, goal_pos_y = goal_pos[0]+1, goal_pos[1]+1
front_cell = env.front_pos
agent_pos_x, agent_pos_y = env.agent_pos[0] + 1, env.agent_pos[1] + 1
# agent_dir = env.agent_dir
if env.agent_dir == 1:
    agent_dir = 3
elif env.agent_dir == 3:
    agent_dir = 1
else:
    agent_dir = env.agent_dir

h,w = env.height, env.width
optim_act_seq = []
q_x = np.ones((4,4,w+2,h+2))*np.inf  # 4 key/door combinations, 4 possible orientations at any point in the grid
# 5 possible actions, 4 key/unlock combinations, 4 orientations for every point in the grid
# key/door combos - 0:no key, door open, 1:key, door open, 2:no key, door closed, 3:key, door closed
# create stage cost matrix; remains static. state space X control space
stg_cost = np.ones((5,4,4,w+2,h+2))
for j in range(num_actions):
    stg_cost[j,:,:,:,:] *= costs[j]

wall_cells = set()
cell_ix_map = {}
c = 1
# set cost of doing anything while being in the door cell to inf unless its open
# stg_cost[:, [2, 3], :, door_pos_x, door_pos_y] = np.inf
# trying to pickup key default set to inf (will be overwritten for cells facing the key)
stg_cost[3,:,:,:,:] = np.inf
# if facing key cell from left, agent must pickup key AND do nothing else
stg_cost[3,2,0,key_pos_x - 1,key_pos_y] = costs[3]
stg_cost[[0,1,2,4],2,0,key_pos_x - 1,key_pos_y] = np.inf
# if facing key cell from right, agent must pickup key AND do nothing else
stg_cost[3,2,2,key_pos_x + 1,key_pos_y] = costs[3]
stg_cost[[0,1,2,4],2,2,key_pos_x + 1,key_pos_y] = np.inf
# if facing key cell from below, agent must pickup key AND do nothing else
stg_cost[3,2,1,key_pos_x,key_pos_y+1] = costs[3]
stg_cost[[0,1,2,4],2,1,key_pos_x,key_pos_y+1] = np.inf
# if facing key cell from above, agent must pickup key AND do nothing else
stg_cost[3,2,3,key_pos_x,key_pos_y-1] = costs[3]
stg_cost[[0,1,2,4],2,3,key_pos_x,key_pos_y-1] = np.inf

# trying to unlock door default set to inf (later the directions facing the door will
# be overwritten to finite costs)
stg_cost[4,:,:,:,:] = np.inf
# find wall cells and add them to wall_cells set with integer indexing
# create terminal cost array based on agent facing goal
for i in range(h):
    for j in range(w):
        cell_ix_map[c] = (j+1,i+1)
        # if door is closed in map, then state when facing door must be key/door = 2/3
        # if facing door from left side and door is closed and agent has key, agent must unlock door
        if j + 1 == door_pos_x - 1 and i + 1 == door_pos_y:
            stg_cost[4, 3, 0, door_pos_x - 1, door_pos_y] = costs[4]
            # move fwd while facing a locked door if carrying or not carrying key is invalid
            stg_cost[0, [2,3], 0, door_pos_x - 1, door_pos_y] = np.inf
            # if is_locked:
            #     stg_cost[:,[0,1],0,door_pos_x - 1, door_pos_y] = np.inf
        # if facing door from below and door is closed and agent has key, agent must unlock door
        # Move fwd facing closed door is invalid
        if j + 1 == door_pos_x and i + 1 == door_pos_y + 1:
            stg_cost[4, 3, 1, door_pos_x, door_pos_y + 1] = costs[4]
            stg_cost[0, [2,3], 1, door_pos_x, door_pos_y + 1] = np.inf
            # if is_locked:
            #     stg_cost[:,[0,1],1,door_pos_x, door_pos_y + 1] = np.inf
        # if facing door from above, must unlock door. Move fwd facing closed door is invalid
        if j + 1 == door_pos_x and i + 1 == door_pos_y - 1:
            stg_cost[4, 3, 3, door_pos_x, door_pos_y - 1] = costs[4]
            stg_cost[0, [2,3], 3, door_pos_x, door_pos_y - 1] = np.inf
            # if is_locked:
            #     stg_cost[:,[0,1],3,door_pos_x, door_pos_y - 1] = np.inf
        # if facing door from right side, must unlock door. Move fwd facing closed door is invalid
        if j + 1 == door_pos_x + 1 and i + 1 == door_pos_y:
            stg_cost[4,3,2,door_pos_x + 1,door_pos_y] = costs[4]
            stg_cost[0,[2,3],2,door_pos_x + 1,door_pos_y] = np.inf
            # if is_locked:
            #     stg_cost[:,[0,1],3,door_pos_x + 1,door_pos_y] = np.inf

        if isinstance(env.grid.get(j,i), Wall):
            wall_cells.add(c)
            wall_x, wall_y = j+1,i+1
            # set cost of being in the wall cell to inf (before taking action)
            stg_cost[:, :, :, wall_x, wall_y] = np.inf
            # cost of being outside grid is also inf (padded cells)
            stg_cost[:,:,:,0,:] = np.inf
            stg_cost[:, :, :, :, 0] = np.inf
            stg_cost[:, :, :, w+1, :] = np.inf
            stg_cost[:, :, :, :, h+1] = np.inf
            # 4 possible ways of moving forward into wall
            # case 1: from below
            if 1 <= wall_y < h:
                # Stage cost of any key/door combo, MF, facing up, from cell below the wall cell = inf
                stg_cost[0,:, 1, wall_x, wall_y + 1] = np.inf
            # case 2: from above
            if 1 <= wall_y - 1 <= h:
                # Stage cost of MF, facing down, from cell above the wall cell = inf
                stg_cost[0,:, 3, wall_x, wall_y - 1] = np.inf
            # case 3: from left
            if 1 <= wall_x - 1 <= w:
                # Stage cost of MF, facing right, from cell to left of the wall cell = inf
                stg_cost[0,:, 0, wall_x - 1, wall_y] = np.inf
            # case 3: from right
            if 1 <= wall_x < w:
                # Stage cost of MF, facing left, from cell to right of the wall cell = inf
                stg_cost[0,:, 2, wall_x + 1, wall_y] = np.inf
        # now check if this cell is not in front of a wall
        else:
            # 0:no key, door open, 1:key, door open, 2:no key, door closed, 3:key, door closed
            # 4 cases where goal is in front of the agent
            # case 1: agent is left of goal
            # if is_locked:
            if j+1 == goal_pos_x - 1 and i+1 == goal_pos_y:
                q_x[0:3,0,j+1,i+1] = 0  # agent direction = 0 is facing right
                # print('left of goal')
            # case 2: agent is right of goal
            if j+1 == goal_pos_x + 1 and i+1 == goal_pos_y:
                q_x[0:3,2,j+1,i+1] = 0
                # print('right of goal')
            # case 3: agent is above goal
            if j+1 == goal_pos_x and i+1 == goal_pos_y - 1:
                q_x[0:3,3,j+1,i+1] = 0
                # print('above goal')
            # case 4: agent is below goal
            if j+1 == goal_pos_x and i+1 == goal_pos_y + 1:
                q_x[0:3,1,j+1,i+1] = 0
                # print('below goal')
            # else:
            #     if j+1 == goal_pos_x - 1 and i+1 == goal_pos_y:
            #         q_x[0,0,j+1,i+1] = 0  # agent direction = 0 is facing right
            #         # print('left of goal')
            #     # case 2: agent is right of goal
            #     if j+1 == goal_pos_x + 1 and i+1 == goal_pos_y:
            #         q_x[0,2,j+1,i+1] = 0
            #         # print('right of goal')
            #     # case 3: agent is above goal
            #     if j+1 == goal_pos_x and i+1 == goal_pos_y - 1:
            #         q_x[0,3,j+1,i+1] = 0
            #         # print('above goal')
            #     # case 4: agent is below goal
            #     if j+1 == goal_pos_x and i+1 == goal_pos_y + 1:
            #         q_x[0,1,j+1,i+1] = 0
            #         # print('below goal')
        c+=1


# motion model
# array with all possible states [x,y,dir,(key/door open close combination)]
door_key_combs = [0,1,2,3]
orientations = [0,1,2,3]
grid_x = list(range(w+2))
grid_y = list(range(h+2))
# order of states is door_comb,orientations,grid_x,grid_y
states = list(itertools.product(door_key_combs,orientations,grid_x[1:-1],grid_y[1:-1]))
# lists for new states caused by applying each action to each state
# action 0 = MF
state_transition_act_0 = []
# action 1 = TL
state_transition_act_1 = []
# action 2 = TR
state_transition_act_2 = []
# action 3 = PK
state_transition_act_3 = []
# action 4 = UD
state_transition_act_4 = []
for ix,state in enumerate(states):
    # recall orientation 0:right, 1:up, 2:left, 3:down
    # transitions under action 0 (MF), action 1 (TL) and action 2 (TR)
    if state[1] == 0:
        state_transition_act_0.append((state[0], state[1], state[2]+1, state[3]))
        state_transition_act_1.append((state[0],1,state[2],state[3]))
        state_transition_act_2.append((state[0], 3, state[2], state[3]))
    # if facing upwards
    if state[1] == 1:
        state_transition_act_0.append((state[0], state[1], state[2], state[3]-1))
        state_transition_act_1.append((state[0],2,state[2],state[3]))
        state_transition_act_2.append((state[0], 0, state[2], state[3]))
    # if facing left
    if state[1] == 2:
        state_transition_act_0.append((state[0], state[1], state[2]-1, state[3]))
        state_transition_act_1.append((state[0],3,state[2],state[3]))
        state_transition_act_2.append((state[0], 1, state[2], state[3]))
    # if facing downwards
    if state[1] == 3:
        state_transition_act_0.append((state[0], state[1], state[2], state[3]+1))
        state_transition_act_1.append((state[0],0,state[2],state[3]))
        state_transition_act_2.append((state[0], 2, state[2], state[3]))
    # transitions under action 3 (PK), action 4 (UD)
    # key/door combos - 0:no key, door open, 1:key, door open, 2:no key, door closed, 3:key, door closed
    if state[0] == 0:
        state_transition_act_3.append((1,state[1],state[2],state[3]))
        state_transition_act_4.append(state)
    if state[0] == 1:
        state_transition_act_3.append(state)
        state_transition_act_4.append(state)
    if state[0] == 2:
        state_transition_act_3.append((3,state[1],state[2],state[3]))
        state_transition_act_4.append((0,state[1],state[2],state[3]))
    if state[0] == 3:
        state_transition_act_3.append(state)
        state_transition_act_4.append((1,state[1],state[2],state[3]))

arr_trans_act0 = np.array(state_transition_act_0).T
arr_trans_act1 = np.array(state_transition_act_1).T
arr_trans_act2 = np.array(state_transition_act_2).T
arr_trans_act3 = np.array(state_transition_act_3).T
arr_trans_act4 = np.array(state_transition_act_4).T

arr_trans_act = np.stack((arr_trans_act0,arr_trans_act1,arr_trans_act2,arr_trans_act3,arr_trans_act4),axis=0)

# main DP loop
# store value fcn at each iteration
V_all = np.zeros((h*w*4*4,4,4,h+2,w+2))
A_all = np.zeros((h*w*4*4,4,4,h+2,w+2))
A_all[0,:,:,:,:] = np.ones((4,4,h+2,w+2))*np.inf
V = deepcopy(q_x)
V_all[0,:,:,:,:] = V
# initialize V_next = q_x so that states at the padded cell positions have value=inf
V_next = np.stack((deepcopy(q_x),deepcopy(q_x),deepcopy(q_x),deepcopy(q_x),deepcopy(q_x)), axis=0)
for i in range(h*w*4*4 - 1):
    # V_{t+1} (x_{t+1}) is the avg val of Value fcn for each x, u combination
    V_next[0,:,:,1:w+1,1:h+1] = np.reshape(V[tuple(arr_trans_act0[0]),tuple(arr_trans_act0[1]),
                               tuple(arr_trans_act0[2]),tuple(arr_trans_act0[3])], (4,4,h,w))
    V_next[1,:,:,1:w+1,1:h+1] = np.reshape(V[tuple(arr_trans_act1[0]),tuple(arr_trans_act1[1]),
                               tuple(arr_trans_act1[2]),tuple(arr_trans_act1[3])], (4,4,h,w))
    V_next[2,:,:,1:w+1,1:h+1] = np.reshape(V[tuple(arr_trans_act2[0]),tuple(arr_trans_act2[1]),
                               tuple(arr_trans_act2[2]),tuple(arr_trans_act2[3])], (4,4,h,w))
    V_next[3,:,:,1:w+1,1:h+1] = np.reshape(V[tuple(arr_trans_act3[0]),tuple(arr_trans_act3[1]),
                               tuple(arr_trans_act3[2]),tuple(arr_trans_act3[3])], (4,4,h,w))
    V_next[4,:,:,1:w+1,1:h+1] = np.reshape(V[tuple(arr_trans_act4[0]),tuple(arr_trans_act4[1]),
                               tuple(arr_trans_act4[2]),tuple(arr_trans_act4[3])], (4,4,h,w))
    Q_t = stg_cost + V_next
    # get optimal value and policy; V changes but V_next structure remains same
    V = Q_t.min(axis=0) # for every state, get the minimum value
    V_all[i+1] = V
    A = Q_t.argmin(axis=0) # for every state, get the optimal action
    # store optimal action
    A_all[i+1] = A

# MF = 0  # Move Forward
# TL = 1  # Turn Left
# TR = 2  # Turn Right
# PK = 3  # Pickup Key
# UD = 4  # Unlock Door
# 0:right, 1:up, 2:left, 3:down

# analyze the optimal policy
# encode the start state based on map args
start = (2,agent_dir,agent_pos_x,agent_pos_y)
start_state_value = V_all[:,start[0],start[1],start[2],start[3]]
# print(start_state_value)
start_reachable_ind = int(start_state_value[start_state_value != np.inf][0])
opt_act_seq = []
curr_state = start
# get the state corresponding to the lowest value at any given time step
for i in range(start_reachable_ind,0,-1):
# for i in range(6):
#     a = np.where(V_all[i] != np.inf)
#     aa = np.stack((a[0], a[1], a[2], a[3]), axis=0).T
    # value_list = []
    # for state in aa:
    #     value_list.append(V_all[i][tuple(state)])
    # print(i,aa)

    # get the optimal action for this time step for this state
    opt_act = int(A_all[i][curr_state])
    print(i,opt_act,curr_state)
    opt_act_seq.append(opt_act)
    # apply motion model to curr_state to get next state
    # first convert the tuple of 4 indices to an int to index the flat state array
    motion_model_ix = curr_state[0]*4*h*w + curr_state[1]*h*w + (curr_state[2]-1)*w + curr_state[3] - 1
    # now pick out the appropriate motion model array corresponding to the optimal action and get the new state
    curr_state = tuple(arr_trans_act[opt_act][:, motion_model_ix])

draw_gif_from_seq(opt_act_seq,load_env(env_path)[0],path='./gif/{x}_updated.gif'.format(x=env_path[-20:-4]))