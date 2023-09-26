import numpy as np
import itertools
from main import load_map, draw_map
import matplotlib.pyplot as plt
import copy
from operator import itemgetter
import time
import matplotlib
from astar import line_intersects_block
matplotlib.use('TkAgg')


class RRTNode(object):
    def __init__(self, id, coord):
        self.id = id
        self.coord = coord
        self.g = np.inf
        self.parent_node = None

class RRT():
    def __init__(self):
        self.optimal_path = []
        self.end_node = None
        self.num_iter = 0
        self.CLOSED = None
        self.id_coord_map = None
        self.id_obj_map = None
        self.nodes = []

    def plan(self, boundary, blocks, start_coord, goal_coord, check_collisions):
        graph = []
        id_coord_map = {}
        coord_id_map = {}
        id_obj_map = {}
        set_coords = set()
        c = 1
        # start node
        start = RRTNode(0, start_coord)
        id_obj_map[start.id] = start
        # start.g = 0
        id_coord_map[start.id] = start.coord
        coord_id_map[tuple(start.coord)] = start.id
        set_coords.add(tuple(start.coord))
        graph.append(start.id)
        iter_ctr = 0
        while True:
            # print(graph)
            if iter_ctr % 10 == 0 and iter_ctr != 0:
                # sample the goal node every 10 iterations to steer A* towards goal faster
                x_rand = goal_coord
                if iter_ctr % 200 == 0:
                    x_nearest_id = GetNearest(goal_coord, graph, id_coord_map)
                    x_nearest = id_coord_map[x_nearest_id]
                    print(iter_ctr, get_distance(x_nearest, goal_coord), x_nearest)
                    if get_distance(x_nearest, goal_coord) <= 1.5*EPSILON:
                        # termination condition - sample goal node and see if can be added to graph
                        node_goal = RRTNode(c, goal_coord)
                        graph.append(node_goal.id)
                        set_coords.add(tuple(goal_coord))
                        node_goal.parent_node = x_nearest_id
                        id_coord_map[node_goal.id] = goal_coord
                        id_obj_map[node_goal.id] = node_goal
                        break
            else:
                x_rand = SampleFree(boundary, blocks)

            x_nearest_id = GetNearest(x_rand, graph, id_coord_map)
            x_nearest = id_coord_map[x_nearest_id]
            x_new = Steer(x_nearest, x_rand)
            if CollisionFree(x_nearest, x_new, blocks):
                # only create node if path to new node is collision free AND node doesn't exist
                if tuple(x_new) not in set_coords:
                    node_x_new = RRTNode(c, x_new)
                    graph.append(node_x_new.id)
                    set_coords.add(tuple(x_new))
                    id_coord_map[node_x_new.id] = x_new
                    node_x_new.parent_node = x_nearest_id
                    id_obj_map[node_x_new.id] = node_x_new
                    c += 1

            iter_ctr += 1

        self.num_iter = iter_ctr
        self.id_coord_map = id_coord_map
        self.id_obj_map = id_obj_map
        self.end_node = node_goal
        self.nodes = list(set_coords)
        return

    def get_optimal_path(self):
        curr = self.end_node
        for i in range(self.num_iter):
            print(curr)
            self.optimal_path.append(curr)
            if curr.id == 0:
                break
            else:
                curr = self.id_obj_map[curr.parent_node]
        return self.optimal_path

def SampleFree(boundary, blocks):
    '''uniformly sample from the free nodes in the environment'''
    resample = 1
    while resample:
        # sample = np.round(np.random.uniform(boundary[0,0:3],boundary[0,3:6]) * (1/STEP_SIZE)) / (1/STEP_SIZE)
        sample = np.round(np.random.uniform(boundary[0,0:3],boundary[0,3:6]),1)
        if point_in_block(sample,blocks):
            resample = 1
        elif not point_in_block(sample,blocks):
            resample = 0
    return sample

def GetNearest(x, G, id_coord_map):
    '''takes in point in grid and graph (list of ids) and returns id of nearest node to x in G'''
    curr_shortest_dist = np.inf
    for g in G:
        g_coord = id_coord_map[g]
        dist = get_distance(x, g_coord)
        if dist < curr_shortest_dist:
            curr_shortest_dist = dist
            nearest_node_id = g
    return nearest_node_id

def Steer(s,target):
    direction = target - s
    normed_dir = direction/np.linalg.norm(direction)
    return np.round(s + EPSILON*normed_dir,1)

def CollisionFree(x,y,blocks):
    return not line_intersects_block(x,y,blocks,EPSILON,COLLISION_STEP_SIZE)

def get_distance(x,y):
    return np.linalg.norm(y-x)

def point_in_block(point,blocks):
    for block in blocks:
        if (
                block[0] <= point[0] <= block[3] and
                block[1] <= point[1] <= block[4] and
                block[2] <= point[2] <= block[5]
        ): return True
    return False

if __name__ == "__main__":
    start_time = time.time()
    EPSILON = 0.1
    # if COLLISION_STEP_SIZE = STEP_SIZE, only the line endpoints are checked
    COLLISION_STEP_SIZE = 0.1
    # GOAL_THRESHOLD = 0.5 * EPSILON
    check_collisions = 1

    mapfile = './maps/single_cube.txt'
    start = np.array([2.3, 2.3, 1.3])
    goal = np.array([7.0, 7.0, 5.5])
    boundary, blocks = load_map(mapfile)

    planner = RRT()
    planner.plan(boundary, blocks, start, goal, check_collisions)
    time_taken = time.time() - start_time
    optimal_path = planner.get_optimal_path()
    path = np.zeros((len(optimal_path), 3))
    for ix, node in enumerate(optimal_path):
        path[-(ix + 1)] = node.coord
    # get total cost of the path taken
    path_cost = np.linalg.norm(path[1:] - path[:-1], axis=1).sum()
    print('Cost of optimal path: ', path_cost)
    print('# of iterations to reach goal: ', planner.num_iter)
    print(time_taken)

    # plot the graph that RRT creates
    arr_nodes = np.array(planner.nodes)
    fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)
    # ax.scatter(0.9,1.9,2.8)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-')
    # # add the expanded node scatter plot
    ax.scatter(arr_nodes[:, 0][::1], arr_nodes[:, 1][::1], arr_nodes[:, 2][::1])
    plt.show()
