from utils import *
import numpy as np
from copy import deepcopy
import itertools
import pickle

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

# directions: +ve is down, right i.e. [0,1] = down, [1,0] = right
# 0:right, 1:up, 2:left, 3:down
# read in random env
env_folder = "./environments/random_envs"
# env, info, env_path = load_random_env(env_folder)
env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder)]
env_path = random.choice(env_list)
print(env_path)
dict_opt_act = {}

with open(env_path, "rb") as f:
    env = pickle.load(f)

info = {"height": env.height,"width": env.width,"init_agent_pos": env.agent_pos,
        "init_agent_dir": env.dir_vec,"door_pos": [],"door_open": []}
for i in range(env.height):
    for j in range(env.width):
        if isinstance(env.grid.get(j, i), Key):
            info["key_pos"] = np.array([j, i])
        elif isinstance(env.grid.get(j, i), Door):
            info["door_pos"].append(np.array([j, i]))
            if env.grid.get(j, i).is_open:
                info["door_open"].append(True)
            else:
                info["door_open"].append(False)
        elif isinstance(env.grid.get(j, i), Goal):
            info["goal_pos"] = np.array([j, i])

# plot_env(env)
door1 = env.grid.get(info["door_pos"][0][0], info["door_pos"][0][1])
door2 = env.grid.get(info["door_pos"][1][0], info["door_pos"][1][1])
door1_pos_x, door1_pos_y = info["door_pos"][0][0] + 1, info["door_pos"][0][1] + 1
door2_pos_x, door2_pos_y = info["door_pos"][1][0] + 1, info["door_pos"][1][1] + 1

key_pos_x, key_pos_y = info["key_pos"][0] + 1, info["key_pos"][1] + 1

# DP implementation
num_actions = 5
# actions: MF,TL,TR,PK,UD = 0,1,2,3,4
costs = {0:1, 1:1, 2:1, 3:1, 4:1}
goal_pos = info['goal_pos']
goal_pos_x, goal_pos_y = goal_pos[0]+1, goal_pos[1]+1
front_cell = env.front_pos
agent_pos_x, agent_pos_y = info['init_agent_pos'][0] + 1, info['init_agent_pos'][1] + 1
if env.agent_dir == 1:
    agent_dir = 3
elif env.agent_dir == 3:
    agent_dir = 1
else:
    agent_dir = env.agent_dir

h,w = env.height, env.width
optim_act_seq = []
q_x = np.ones((8,4,3,3,w+2,h+2))*np.inf  # 8 key/door combinations, 4 possible orientations at any point in the grid
# 5 possible actions, 4 key/unlock combinations, 4 orientations for every point in the grid
# key/door combos - 0:no key, door open, 1:key, door open, 2:no key, door closed, 3:key, door closed
# create stage cost matrix; remains static. state space X control space
stg_cost = np.ones((5,8,4,3,3,w+2,h+2))
for j in range(num_actions):
    stg_cost[j,:,:,:,:,:,:] *= costs[j]

wall_cells = set()
cell_ix_map = {}
c = 1

if key_pos_x == 2 and key_pos_y == 2:
    key_pos_ix = 0
elif key_pos_x == 3 and key_pos_y == 4:
    key_pos_ix = 1
elif key_pos_x == 2 and key_pos_y == 7:
    key_pos_ix = 2

key_pos1_x, key_pos1_y = 2,2
key_pos2_x, key_pos2_y = 3,4
key_pos3_x, key_pos3_y = 2,7
# in stage costs, the first 3 dim is the key pos, and second 3 dim is the possible goal positions
# trying to pickup key default set to inf (will be overwritten for cells facing the key)
stg_cost[3,:,:,:,:,:,:] = np.inf
# allow agent to pickup key if either door is locked and agent doesn't already have the key
# if facing key cell from left and both doors locked, agent must pickup key AND do nothing else
stg_cost[3,[0,2,6],0,0,:,key_pos1_x - 1,key_pos1_y] = costs[3]
stg_cost[[0,1,2,4],2,0,0,:,key_pos1_x - 1,key_pos1_y] = np.inf
# if facing key cell from right and both doors locked, agent must pickup key AND do nothing else
stg_cost[3,[0,2,6],2,0,:,key_pos1_x + 1,key_pos1_y] = costs[3]
stg_cost[[0,1,2,4],2,2,0,:,key_pos1_x + 1,key_pos1_y] = np.inf
# if facing key cell from below and both doors locked, agent must pickup key AND do nothing else
stg_cost[3,[0,2,6],1,0,:,key_pos1_x,key_pos1_y+1] = costs[3]
stg_cost[[0,1,2,4],2,1,0,:,key_pos1_x,key_pos1_y+1] = np.inf
# if facing key cell from above and both doors locked, agent must pickup key AND do nothing else
stg_cost[3,[0,2,6],3,0,:,key_pos1_x,key_pos1_y-1] = costs[3]
stg_cost[[0,1,2,4],2,3,0,:,key_pos1_x,key_pos1_y-1] = np.inf

#key pos 2
# allow agent to pickup key if either door is locked and agent doesn't already have the key
# if facing key cell from left and both doors locked, agent must pickup key AND do nothing else
stg_cost[3,[0,2,6],0,1,:,key_pos2_x - 1,key_pos2_y] = costs[3]
stg_cost[[0,1,2,4],2,0,1,:,key_pos2_x - 1,key_pos2_y] = np.inf
# if facing key cell from right and both doors locked, agent must pickup key AND do nothing else
stg_cost[3,[0,2,6],2,1,:,key_pos2_x + 1,key_pos2_y] = costs[3]
stg_cost[[0,1,2,4],2,2,1,:,key_pos2_x + 1,key_pos2_y] = np.inf
# if facing key cell from below and both doors locked, agent must pickup key AND do nothing else
stg_cost[3,[0,2,6],1,1,:,key_pos2_x,key_pos2_y+1] = costs[3]
stg_cost[[0,1,2,4],2,1,1,:,key_pos2_x,key_pos2_y+1] = np.inf
# if facing key cell from above and both doors locked, agent must pickup key AND do nothing else
stg_cost[3,[0,2,6],3,1,:,key_pos2_x,key_pos2_y-1] = costs[3]
stg_cost[[0,1,2,4],2,3,1,:,key_pos2_x,key_pos2_y-1] = np.inf

# key pos 3
# allow agent to pickup key if either door is locked and agent doesn't already have the key
# if facing key cell from left and both doors locked, agent must pickup key AND do nothing else
stg_cost[3,[0,2,6],0,2,:,key_pos3_x - 1,key_pos3_y] = costs[3]
stg_cost[[0,1,2,4],2,0,2,:,key_pos3_x - 1,key_pos3_y] = np.inf
# if facing key cell from right and both doors locked, agent must pickup key AND do nothing else
stg_cost[3,[0,2,6],2,2,:,key_pos3_x + 1,key_pos3_y] = costs[3]
stg_cost[[0,1,2,4],2,2,2,:,key_pos3_x + 1,key_pos3_y] = np.inf
# if facing key cell from below and both doors locked, agent must pickup key AND do nothing else
stg_cost[3,[0,2,6],1,2,:,key_pos3_x,key_pos3_y+1] = costs[3]
stg_cost[[0,1,2,4],2,1,2,:,key_pos3_x,key_pos3_y+1] = np.inf
# if facing key cell from above and both doors locked, agent must pickup key AND do nothing else
stg_cost[3,[0,2,6],3,2,:,key_pos3_x,key_pos3_y-1] = costs[3]
stg_cost[[0,1,2,4],2,3,2,:,key_pos3_x,key_pos3_y-1] = np.inf


possible_goal_pos = [(6,2),(7,4),(6,7)]
q_x[0:8, 0,:,0, 5, 2] = 0  # agent direction = 0 is facing right
q_x[0:8, 2,:,0, 7, 2] = 0
q_x[0:8, 3,:,0, 6, 1] = 0
q_x[0:8, 1,:,0, 6, 3] = 0

q_x[0:8, 0,:,1, 6, 4] = 0
q_x[0:8, 2,:,1, 8, 4] = 0
q_x[0:8, 3,:,1, 7, 3] = 0
q_x[0:8, 1,:,1, 7, 5] = 0

q_x[0:8, 0,:,2, 5, 7] = 0
q_x[0:8, 2,:,2, 7, 7] = 0
q_x[0:8, 3,:,2, 6, 6] = 0
q_x[0:8, 1,:,2, 6, 8] = 0

# terminal costs
if goal_pos_x == 6 and goal_pos_y == 2:
    goal_ix = 0
elif goal_pos_x == 7 and goal_pos_y == 4:
    goal_ix = 1
elif goal_pos_x == 6 and goal_pos_y == 7:
    goal_ix = 2

print(key_pos_ix, goal_ix)
# trying to unlock door default set to inf (later the directions facing the door will
# be overwritten to finite costs)
stg_cost[4,:,:,:,:,:,:] = np.inf
# find wall cells and add them to wall_cells set with integer indexing
# create terminal cost array based on agent facing goal
for i in range(h):
    for j in range(w):
        cell_ix_map[c] = (j+1,i+1)
        # if facing door 1 from left side and door 1 is closed and agent has key, agent must unlock door
        if j + 1 == door1_pos_x - 1 and i + 1 == door1_pos_y:
            # door 1 must be closed but door 2 can be open or closed
            stg_cost[4, [3,7], 0,:,:, door1_pos_x - 1, door1_pos_y] = costs[4]
            # move fwd while facing a locked door if carrying or not carrying key is invalid
            stg_cost[0, [2,3,6,7], 0,:,:, door1_pos_x - 1, door1_pos_y] = np.inf
        # if facing door 1 from right side, must unlock door. Move fwd facing closed door is invalid
        if j + 1 == door1_pos_x + 1 and i + 1 == door1_pos_y:
            stg_cost[4,[3,7],2,:,:,door1_pos_x + 1,door1_pos_y] = costs[4]
            stg_cost[0,[2,3,6,7],2,:,:,door1_pos_x + 1,door1_pos_y] = np.inf

        # door2
        if j + 1 == door2_pos_x - 1 and i + 1 == door2_pos_y:
            stg_cost[4, [1,3], 0,:,:, door2_pos_x - 1, door2_pos_y] = costs[4]
            # move fwd while facing a locked door if carrying or not carrying key is invalid
            stg_cost[0, [0,1,2,3], 0, :,:,door2_pos_x - 1, door2_pos_y] = np.inf
        # if facing door 2 from right side, must unlock door. Move fwd facing closed door is invalid
        if j + 1 == door2_pos_x + 1 and i + 1 == door2_pos_y:
            stg_cost[4,[1,3],2,:,:,door2_pos_x + 1,door2_pos_y] = costs[4]
            stg_cost[0,[0,1,2,3],2,:,:,door2_pos_x + 1,door2_pos_y] = np.inf

        if isinstance(env.grid.get(j,i), Wall):
            wall_cells.add(c)
            wall_x, wall_y = j+1,i+1
            # set cost of being in the wall cell to inf (before taking action)
            stg_cost[:, :, :,:,:, wall_x, wall_y] = np.inf
            # cost of being outside grid is also inf (padded cells)
            stg_cost[:,:,:,:,:,0,:] = np.inf
            stg_cost[:, :, :, :,:,:, 0] = np.inf
            stg_cost[:, :, :,:,:, w+1, :] = np.inf
            stg_cost[:, :, :, :,:,:, h+1] = np.inf
            # 4 possible ways of moving forward into wall
            # case 1: from below
            if 1 <= wall_y < h:
                # Stage cost of any key/door combo, MF, facing up, from cell below the wall cell = inf
                stg_cost[0,:, 1, :,:,wall_x, wall_y + 1] = np.inf
            # case 2: from above
            if 1 <= wall_y - 1 <= h:
                # Stage cost of MF, facing down, from cell above the wall cell = inf
                stg_cost[0,:, 3, :,:,wall_x, wall_y - 1] = np.inf
            # case 3: from left
            if 1 <= wall_x - 1 <= w:
                # Stage cost of MF, facing right, from cell to left of the wall cell = inf
                stg_cost[0,:, 0, :,:, wall_x - 1, wall_y] = np.inf
            # case 3: from right
            if 1 <= wall_x < w:
                # Stage cost of MF, facing left, from cell to right of the wall cell = inf
                stg_cost[0,:, 2, :,:, wall_x + 1, wall_y] = np.inf
        c+=1


# motion model
# array with all possible states [x,y,dir,(key/door open close combination)]

door_key_combs = [0,1,2,3,4,5,6,7]
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
    # key/door combos - 0:no key, door open, 1:key, door open, 2:no key, door closed, 3:key, door closed
    if state[0] == 0:
        state_transition_act_3.append((1,state[1],state[2],state[3]))
        state_transition_act_4.append(state)
    if state[0] == 1:
        state_transition_act_3.append(state)
        state_transition_act_4.append((5,state[1],state[2],state[3]))
    if state[0] == 2:
        state_transition_act_3.append((3,state[1],state[2],state[3]))
        state_transition_act_4.append(state)
    if state[0] == 3:
        state_transition_act_3.append(state)
        state_transition_act_4.append((5,state[1],state[2],state[3]))
    if state[0] == 4:
        state_transition_act_3.append((5, state[1], state[2], state[3]))
        state_transition_act_4.append(state)
    if state[0] == 5:
        state_transition_act_3.append(state)
        state_transition_act_4.append(state)
    if state[0] == 6:
        state_transition_act_3.append((7, state[1], state[2], state[3]))
        state_transition_act_4.append(state)
    if state[0] == 7:
        state_transition_act_3.append(state)
        state_transition_act_4.append((5, state[1], state[2], state[3]))

# door_key_combs = [0,1,2,3,4,5,6,7]
# orientations = [0,1,2,3]
# key_positions = [0,1,2]
# goal_positions = [0,1,2]
# grid_x = list(range(w+2))
# grid_y = list(range(h+2))
# # order of states is door_comb,orientations,grid_x,grid_y
# states = list(itertools.product(door_key_combs,orientations,key_positions,goal_positions,grid_x[1:-1],grid_y[1:-1]))
# # lists for new states caused by applying each action to each state
# # action 0 = MF
# state_transition_act_0 = []
# # action 1 = TL
# state_transition_act_1 = []
# # action 2 = TR
# state_transition_act_2 = []
# # action 3 = PK
# state_transition_act_3 = []
# # action 4 = UD
# state_transition_act_4 = []
# for ix,state in enumerate(states):
#     # recall orientation 0:right, 1:up, 2:left, 3:down
#     # transitions under action 0 (MF), action 1 (TL) and action 2 (TR)
#     if state[1] == 0:
#         state_transition_act_0.append((state[0], state[1], state[2],state[3],state[4]+1, state[5]))
#         state_transition_act_1.append((state[0],1,state[2],state[3],state[4],state[5]))
#         state_transition_act_2.append((state[0], 3, state[2],state[3],state[4], state[5]))
#     # if facing upwards
#     if state[1] == 1:
#         state_transition_act_0.append((state[0], state[1],state[2],state[3], state[4], state[5]-1))
#         state_transition_act_1.append((state[0],2,state[2],state[3],state[4],state[5]))
#         state_transition_act_2.append((state[0], 0, state[2],state[3], state[4], state[5]))
#     # if facing left
#     if state[1] == 2:
#         state_transition_act_0.append((state[0], state[1], state[2],state[3], state[4]-1, state[5]))
#         state_transition_act_1.append((state[0],3,state[2],state[3],state[4],state[5]))
#         state_transition_act_2.append((state[0], 1, state[2],state[3], state[4], state[5]))
#     # if facing downwards
#     if state[1] == 3:
#         state_transition_act_0.append((state[0], state[1], state[2],state[3],state[4], state[5]+1))
#         state_transition_act_1.append((state[0],0,state[2],state[3],state[4],state[5]))
#         state_transition_act_2.append((state[0], 2, state[2],state[3],state[4], state[5]))
#     # key/door combos - 0:no key, door open, 1:key, door open, 2:no key, door closed, 3:key, door closed
#     if state[0] == 0:
#         state_transition_act_3.append((1,state[1],state[2],state[3],state[4],state[5]))
#         state_transition_act_4.append(state)
#     if state[0] == 1:
#         state_transition_act_3.append(state)
#         state_transition_act_4.append((5,state[1],state[2],state[3],state[4],state[5]))
#     if state[0] == 2:
#         state_transition_act_3.append((3,state[1],state[2],state[3],state[4],state[5]))
#         state_transition_act_4.append(state)
#     if state[0] == 3:
#         state_transition_act_3.append(state)
#         state_transition_act_4.append((5,state[1],state[2],state[3],state[4],state[5]))
#     if state[0] == 4:
#         state_transition_act_3.append((5, state[1], state[2],state[3],state[4], state[5]))
#         state_transition_act_4.append(state)
#     if state[0] == 5:
#         state_transition_act_3.append(state)
#         state_transition_act_4.append(state)
#     if state[0] == 6:
#         state_transition_act_3.append((7, state[1], state[2],state[3],state[4], state[5]))
#         state_transition_act_4.append(state)
#     if state[0] == 7:
#         state_transition_act_3.append(state)
#         state_transition_act_4.append((5, state[1], state[2],state[3],state[4], state[5]))

arr_trans_act0 = np.array(state_transition_act_0).T
arr_trans_act1 = np.array(state_transition_act_1).T
arr_trans_act2 = np.array(state_transition_act_2).T
arr_trans_act3 = np.array(state_transition_act_3).T
arr_trans_act4 = np.array(state_transition_act_4).T

arr_trans_act = np.stack((arr_trans_act0,arr_trans_act1,arr_trans_act2,arr_trans_act3,arr_trans_act4),axis=0)

# main DP loop
# store value fcn at each iteration
V_all = np.zeros((h*w*8*4,8,4,3,3,h+2,w+2))
A_all = np.zeros((h*w*8*4,8,4,3,3,h+2,w+2))
A_all[0,:,:,:,:,:,:] = np.ones((8,4,3,3,h+2,w+2))*np.inf
V = deepcopy(q_x)
V_all[0,:,:,:,:,:,:] = V
# initialize V_next = q_x so that states at the padded cell positions have value=inf
V_next = np.stack((deepcopy(q_x),deepcopy(q_x),deepcopy(q_x),deepcopy(q_x),deepcopy(q_x)), axis=0)
# for i in range(h*w*8*4*3*3 - 1):
for i in range(h*w*8*4 - 1):
    # V_{t+1} (x_{t+1}) is the avg val of Value fcn for each x, u combination
    V_next[0,:,:,0,0,1:w+1,1:h+1] = np.reshape(V[tuple(arr_trans_act0[0]),tuple(arr_trans_act0[1]),
                               0,0,tuple(arr_trans_act0[2]),tuple(arr_trans_act0[3])], (8,4,h,w))
    V_next[1,:,:,0,0,1:w+1,1:h+1] = np.reshape(V[tuple(arr_trans_act1[0]),tuple(arr_trans_act1[1]),
                               0,0,tuple(arr_trans_act1[2]),tuple(arr_trans_act1[3])], (8,4,h,w))
    V_next[2,:,:,0,0,1:w+1,1:h+1] = np.reshape(V[tuple(arr_trans_act2[0]),tuple(arr_trans_act2[1]),
                               0,0,tuple(arr_trans_act2[2]),tuple(arr_trans_act2[3])], (8,4,h,w))
    V_next[3,:,:,0,0,1:w+1,1:h+1] = np.reshape(V[tuple(arr_trans_act3[0]),tuple(arr_trans_act3[1]),
                               0,0,tuple(arr_trans_act3[2]),tuple(arr_trans_act3[3])], (8,4,h,w))
    V_next[4,:,:,0,0,1:w+1,1:h+1] = np.reshape(V[tuple(arr_trans_act4[0]),tuple(arr_trans_act4[1]),
                               0,0,tuple(arr_trans_act4[2]),tuple(arr_trans_act4[3])], (8,4,h,w))

    V_next[0, :, :, 0, 1, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act0[0]), tuple(arr_trans_act0[1]),
                                                           0, 1, tuple(arr_trans_act0[2]), tuple(arr_trans_act0[3])],
                                                         (8, 4, h, w))
    V_next[1, :, :, 0, 1, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act1[0]), tuple(arr_trans_act1[1]),
                                                           0, 1, tuple(arr_trans_act1[2]), tuple(arr_trans_act1[3])],
                                                         (8, 4, h, w))
    V_next[2, :, :, 0, 1, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act2[0]), tuple(arr_trans_act2[1]),
                                                           0, 1, tuple(arr_trans_act2[2]), tuple(arr_trans_act2[3])],
                                                         (8, 4, h, w))
    V_next[3, :, :, 0, 1, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act3[0]), tuple(arr_trans_act3[1]),
                                                           0, 1, tuple(arr_trans_act3[2]), tuple(arr_trans_act3[3])],
                                                         (8, 4, h, w))
    V_next[4, :, :, 0, 1, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act4[0]), tuple(arr_trans_act4[1]),
                                                           0, 1, tuple(arr_trans_act4[2]), tuple(arr_trans_act4[3])],
                                                         (8, 4, h, w))

    V_next[0, :, :, 0, 2, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act0[0]), tuple(arr_trans_act0[1]),
                                                           0, 2, tuple(arr_trans_act0[2]), tuple(arr_trans_act0[3])],
                                                         (8, 4, h, w))
    V_next[1, :, :, 0, 2, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act1[0]), tuple(arr_trans_act1[1]),
                                                           0, 2, tuple(arr_trans_act1[2]), tuple(arr_trans_act1[3])],
                                                         (8, 4, h, w))
    V_next[2, :, :, 0, 2, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act2[0]), tuple(arr_trans_act2[1]),
                                                           0, 2, tuple(arr_trans_act2[2]), tuple(arr_trans_act2[3])],
                                                         (8, 4, h, w))
    V_next[3, :, :, 0, 2, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act3[0]), tuple(arr_trans_act3[1]),
                                                           0, 2, tuple(arr_trans_act3[2]), tuple(arr_trans_act3[3])],
                                                         (8, 4, h, w))
    V_next[4, :, :, 0, 2, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act4[0]), tuple(arr_trans_act4[1]),
                                                           0, 2, tuple(arr_trans_act4[2]), tuple(arr_trans_act4[3])],
                                                         (8, 4, h, w))

    V_next[0, :, :, 1, 0, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act0[0]), tuple(arr_trans_act0[1]),
                                                           1, 0, tuple(arr_trans_act0[2]), tuple(arr_trans_act0[3])],
                                                         (8, 4, h, w))
    V_next[1, :, :, 1, 0, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act1[0]), tuple(arr_trans_act1[1]),
                                                           1, 0, tuple(arr_trans_act1[2]), tuple(arr_trans_act1[3])],
                                                         (8, 4, h, w))
    V_next[2, :, :, 1, 0, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act2[0]), tuple(arr_trans_act2[1]),
                                                           1, 0, tuple(arr_trans_act2[2]), tuple(arr_trans_act2[3])],
                                                         (8, 4, h, w))
    V_next[3, :, :, 1, 0, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act3[0]), tuple(arr_trans_act3[1]),
                                                           1, 0, tuple(arr_trans_act3[2]), tuple(arr_trans_act3[3])],
                                                         (8, 4, h, w))
    V_next[4, :, :, 1, 0, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act4[0]), tuple(arr_trans_act4[1]),
                                                           1, 0, tuple(arr_trans_act4[2]), tuple(arr_trans_act4[3])],
                                                         (8, 4, h, w))

    V_next[0, :, :, 1, 1, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act0[0]), tuple(arr_trans_act0[1]),
                                                           1, 1, tuple(arr_trans_act0[2]), tuple(arr_trans_act0[3])],
                                                         (8, 4, h, w))
    V_next[1, :, :, 1, 1, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act1[0]), tuple(arr_trans_act1[1]),
                                                           1, 1, tuple(arr_trans_act1[2]), tuple(arr_trans_act1[3])],
                                                         (8, 4, h, w))
    V_next[2, :, :, 1, 1, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act2[0]), tuple(arr_trans_act2[1]),
                                                           1, 1, tuple(arr_trans_act2[2]), tuple(arr_trans_act2[3])],
                                                         (8, 4, h, w))
    V_next[3, :, :, 1, 1, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act3[0]), tuple(arr_trans_act3[1]),
                                                           1, 1, tuple(arr_trans_act3[2]), tuple(arr_trans_act3[3])],
                                                         (8, 4, h, w))
    V_next[4, :, :, 1, 1, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act4[0]), tuple(arr_trans_act4[1]),
                                                           1, 1, tuple(arr_trans_act4[2]), tuple(arr_trans_act4[3])],
                                                         (8, 4, h, w))

    V_next[0, :, :, 1, 2, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act0[0]), tuple(arr_trans_act0[1]),
                                                           1, 2, tuple(arr_trans_act0[2]), tuple(arr_trans_act0[3])],
                                                         (8, 4, h, w))
    V_next[1, :, :, 1, 2, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act1[0]), tuple(arr_trans_act1[1]),
                                                           1, 2, tuple(arr_trans_act1[2]), tuple(arr_trans_act1[3])],
                                                         (8, 4, h, w))
    V_next[2, :, :, 1, 2, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act2[0]), tuple(arr_trans_act2[1]),
                                                           1, 2, tuple(arr_trans_act2[2]), tuple(arr_trans_act2[3])],
                                                         (8, 4, h, w))
    V_next[3, :, :, 1, 2, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act3[0]), tuple(arr_trans_act3[1]),
                                                           1, 2, tuple(arr_trans_act3[2]), tuple(arr_trans_act3[3])],
                                                         (8, 4, h, w))
    V_next[4, :, :, 1, 2, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act4[0]), tuple(arr_trans_act4[1]),
                                                           1, 2, tuple(arr_trans_act4[2]), tuple(arr_trans_act4[3])],
                                                         (8, 4, h, w))

    V_next[0, :, :, 2, 0, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act0[0]), tuple(arr_trans_act0[1]),
                                                           2, 0, tuple(arr_trans_act0[2]), tuple(arr_trans_act0[3])],
                                                         (8, 4, h, w))
    V_next[1, :, :, 2, 0, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act1[0]), tuple(arr_trans_act1[1]),
                                                           2, 0, tuple(arr_trans_act1[2]), tuple(arr_trans_act1[3])],
                                                         (8, 4, h, w))
    V_next[2, :, :, 2, 0, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act2[0]), tuple(arr_trans_act2[1]),
                                                           2, 0, tuple(arr_trans_act2[2]), tuple(arr_trans_act2[3])],
                                                         (8, 4, h, w))
    V_next[3, :, :, 2, 0, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act3[0]), tuple(arr_trans_act3[1]),
                                                           2, 0, tuple(arr_trans_act3[2]), tuple(arr_trans_act3[3])],
                                                         (8, 4, h, w))
    V_next[4, :, :, 2, 0, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act4[0]), tuple(arr_trans_act4[1]),
                                                           2, 0, tuple(arr_trans_act4[2]), tuple(arr_trans_act4[3])],
                                                         (8, 4, h, w))

    V_next[0, :, :, 2, 1, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act0[0]), tuple(arr_trans_act0[1]),
                                                           2, 1, tuple(arr_trans_act0[2]), tuple(arr_trans_act0[3])],
                                                         (8, 4, h, w))
    V_next[1, :, :, 2, 1, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act1[0]), tuple(arr_trans_act1[1]),
                                                           2, 1, tuple(arr_trans_act1[2]), tuple(arr_trans_act1[3])],
                                                         (8, 4, h, w))
    V_next[2, :, :, 2, 1, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act2[0]), tuple(arr_trans_act2[1]),
                                                           2, 1, tuple(arr_trans_act2[2]), tuple(arr_trans_act2[3])],
                                                         (8, 4, h, w))
    V_next[3, :, :, 2, 1, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act3[0]), tuple(arr_trans_act3[1]),
                                                           2, 1, tuple(arr_trans_act3[2]), tuple(arr_trans_act3[3])],
                                                         (8, 4, h, w))
    V_next[4, :, :, 2, 1, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act4[0]), tuple(arr_trans_act4[1]),
                                                           2, 1, tuple(arr_trans_act4[2]), tuple(arr_trans_act4[3])],
                                                         (8, 4, h, w))

    V_next[0, :, :, 2, 2, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act0[0]), tuple(arr_trans_act0[1]),
                                                           2, 2, tuple(arr_trans_act0[2]), tuple(arr_trans_act0[3])],
                                                         (8, 4, h, w))
    V_next[1, :, :, 2, 2, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act1[0]), tuple(arr_trans_act1[1]),
                                                           2, 2, tuple(arr_trans_act1[2]), tuple(arr_trans_act1[3])],
                                                         (8, 4, h, w))
    V_next[2, :, :, 2, 2, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act2[0]), tuple(arr_trans_act2[1]),
                                                           2, 2, tuple(arr_trans_act2[2]), tuple(arr_trans_act2[3])],
                                                         (8, 4, h, w))
    V_next[3, :, :, 2, 2, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act3[0]), tuple(arr_trans_act3[1]),
                                                           2, 2, tuple(arr_trans_act3[2]), tuple(arr_trans_act3[3])],
                                                         (8, 4, h, w))
    V_next[4, :, :, 2, 2, 1:w + 1, 1:h + 1] = np.reshape(V[tuple(arr_trans_act4[0]), tuple(arr_trans_act4[1]),
                                                           2, 2, tuple(arr_trans_act4[2]), tuple(arr_trans_act4[3])],
                                                         (8, 4, h, w))


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
if door1.is_open and door2.is_open:
    state0 = 4
elif door1.is_locked and door2.is_open:
    state0 = 6
elif door1.is_open and door2.is_locked:
    state0 = 0
elif door1.is_locked and door2.is_locked:
    state0 = 2

start = (state0,key_pos_ix,goal_ix,1,4,6)
start_state_value = V_all[:,start[0],start[1],start[2],start[3],start[4],start[5]]
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
    motion_model_ix = curr_state[0]*4*h*w + curr_state[1]*h*w + (curr_state[4]-1)*w + curr_state[5] - 1
    # now pick out the appropriate motion model array corresponding to the optimal action and get the new state
    curr_state = np.zeros(6)
    act_ind = arr_trans_act[opt_act][:, motion_model_ix]
    curr_state[0] = act_ind[0]
    curr_state[1] = act_ind[1]
    curr_state[4] = act_ind[2]
    curr_state[5] = act_ind[3]
    curr_state[2] = key_pos_ix
    curr_state[3] = goal_ix
    curr_state = tuple(curr_state.astype(int))
print(opt_act_seq)
# print(dict_opt_act)
# draw_gif_from_seq(opt_act_seq,load_env(env_path)[0],path='./gif_partb/{x}.gif'.format(x=env_path[-9:-4]))
# dict_letter = {}
# dict_act_let = {0:'MF',1:'TL',2:'TR',3:'PK',4:'UD'}
# for k,v in dict_opt_act.items():
#     dict_letter[k] = []
#     for act in v:
#         dict_letter[k].append(dict_act_let[act])