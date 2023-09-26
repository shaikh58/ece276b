# priority queue for OPEN list
from pqdict import pqdict
import numpy as np
import itertools
from main import load_map, draw_map
import matplotlib.pyplot as plt
from operator import itemgetter
import time
import matplotlib
matplotlib.use('TkAgg')

class AStarNode(object):
    def __init__(self, id, coord, hval):
        self.id = id
        self.coord = coord
        self.g = np.inf
        self.h = hval
        self.parent_node = None
        self.parent_action = None
        self.closed = False


def get_heuristics(arr_nodes, goal):
    return np.round(np.linalg.norm((goal - arr_nodes),ord=1, axis=1),2)


def get_edge_costs(curr, next_nodes):
    return np.round(np.linalg.norm((curr - next_nodes), axis=1),2)


class AStar(object):
    def __init__(self):
        self.optimal_path = []
        self.end_node = None
        self.num_iter = 0
        self.CLOSED = None
        self.id_obj_map = None

    # @staticmethod
    def plan(self, boundary, blocks, start_coord, goal_coord, check_collisions, epsilon=1):
        # Initialize the graph and open list
        OPEN = pqdict()
        CLOSED = set()
        id_obj_map = {}
        coord_id_map = {}
        bxmin,bymin,bzmin,bxmax,bymax,bzmax = boundary[0][0],boundary[0][1],boundary[0][2],\
                                              boundary[0][3],boundary[0][4],boundary[0][5]
        c = 1
        # start node
        start = AStarNode(0, start_coord, hval=np.linalg.norm(start_coord - goal_coord))
        start.g = 0
        id_obj_map[start.id] = start
        coord_id_map[tuple(start.coord)] = start.id
        OPEN[start.id] = start.g + epsilon * start.h
        # define offsets to find child nodes of each node
        offsets = np.round(np.array(list(itertools.product([0, 1, -1], repeat=3))[1:])*STEP_SIZE,2)
        num_children = 26
        iter_ctr = 0
        # while iter_ctr <= 20:
        while True:
            # print('New iteration starting')
            curr_node_id = OPEN.pop()
            CLOSED.add(curr_node_id)
            # get the node object corresponding to the id
            curr_node = id_obj_map[curr_node_id]
            # print('Node expanded from open list: ', curr_node_id, curr_node.h, curr_node.g)
            print(iter_ctr, len(OPEN), curr_node.coord, goal_coord)
            if np.linalg.norm(curr_node.coord - goal_coord) <= GOAL_THRESHOLD:
                print('Goal reached')
                self.end_node = curr_node
                self.num_iter = iter_ctr
                break
            arr_child_coords = np.round(curr_node.coord + offsets,2)
            arr_edge_costs = get_edge_costs(curr_node.coord, arr_child_coords)
            arr_child_heuristics = get_heuristics(arr_child_coords, goal_coord)
            for j in range(num_children):
                # check that the child node is inside the boundaries
                if bxmin <= arr_child_coords[j][0] <= bxmax and \
                        bymin <= arr_child_coords[j][1] <= bymax and \
                        bzmin <= arr_child_coords[j][2] <= bzmax:
                    if check_collisions:
                        # if child node collides with obstacle, skip it
                        if line_intersects_block(curr_node.coord,arr_child_coords[j],blocks, STEP_SIZE,COLLISION_STEP_SIZE):
                            # print('child node collides with obstacle: ',j,curr_node.coord,arr_child_coords[j])
                            continue
                        else:
                            # check if this child already has a node associated with it (O(1))
                            if tuple(arr_child_coords[j]) not in coord_id_map:
                                # create node if it doesn't exist
                                node = AStarNode(c, arr_child_coords[j], arr_child_heuristics[j])
                                id_obj_map[node.id] = node
                                coord_id_map[tuple(node.coord)] = node.id
                                c += 1
                            # but if child node already exists, retrieve it
                            else:
                                node_id = coord_id_map[tuple(arr_child_coords[j])]
                                node = id_obj_map[node_id]
                            # print('Checking child with id: ', node.coord)#, curr_node.coord,arr_edge_costs[j],arr_child_heuristics[j])
                            # ensure the child not is not in CLOSED before correcting label
                            if node.id not in CLOSED:
                                if node.g > np.round(curr_node.g + arr_edge_costs[j],2):
                                    node.g = np.round(curr_node.g + arr_edge_costs[j],2)
                                    node.parent_node = curr_node
                                    # if child not in OPEN, add to open else update value
                                    OPEN[node.id] = np.round(node.g + epsilon * node.h,2)
                    # if check_collisions is switched off, don't check
                    else:
                        # check if this child already has a node associated with it (O(1))
                        if tuple(arr_child_coords[j]) not in coord_id_map:
                            # create node if it doesn't exist
                            node = AStarNode(c, arr_child_coords[j], arr_child_heuristics[j])
                            id_obj_map[node.id] = node
                            coord_id_map[tuple(node.coord)] = node.id
                            c += 1
                        # but if child node already exists, retrieve it
                        else:
                            node_id = coord_id_map[tuple(arr_child_coords[j])]
                            node = id_obj_map[node_id]
                        # print('Checking child with id: ', node.coord)#, curr_node.coord,arr_edge_costs[j],arr_child_heuristics[j])
                        # ensure the child not is not in CLOSED before correcting label
                        if node.id not in CLOSED:
                            if node.g > np.round(curr_node.g + arr_edge_costs[j],2):
                                node.g = np.round(curr_node.g + arr_edge_costs[j],2)
                                node.parent_node = curr_node
                                OPEN[node.id] = np.round(node.g + epsilon * node.h,2)

            iter_ctr+=1
        self.CLOSED = CLOSED
        self.id_obj_map = id_obj_map
        return

    def get_optimal_path(self):
        curr = self.end_node
        for i in range(self.num_iter):
            self.optimal_path.append(curr)
            if curr.id == 0:
                break
            else:
                curr = curr.parent_node
        return self.optimal_path

def line_intersects_block(line_start, line_end, blocks, step,collision_step_size):
  for block in blocks:
    # Check if the line intersects any of the block's faces
    vec = line_end - line_start
    vec = (vec > 0).astype(int)
    for t in range(int(step/collision_step_size)+1):
      pos = line_start + collision_step_size*t*vec
      # print(pos)
      if (
              block[0] <= pos[0] <= block[3] and
              block[1] <= pos[1] <= block[4] and
              block[2] <= pos[2] <= block[5]
      ): return True
  return False

if __name__ == "__main__":
    start_time = time.time()
    # STEP_SIZE should be half the magnitude of the length of the thinnest obstacle dimension
    # e.g. if min obstacle thickness = 0.1, step size must be 0.05
    STEP_SIZE = 0.5
    # if COLLISION_STEP_SIZE = STEP_SIZE, only the line endpoints are checked
    COLLISION_STEP_SIZE = 0.1
    GOAL_THRESHOLD = 0.8 * STEP_SIZE
    check_collisions = 1

    mapfile = './maps/single_cube.txt'
    start = np.array([2.3, 2.3, 1.3])
    goal = np.array([7.0, 7.0, 5.5])
    boundary, blocks = load_map(mapfile)

    planner = AStar()
    planner.plan(boundary, blocks, start, goal, check_collisions,epsilon=1)
    time_taken = time.time() - start_time
    optimal_path = planner.get_optimal_path()
    path = np.zeros((len(optimal_path),3))
    for ix,node in enumerate(optimal_path):
        path[-(ix+1)] = node.coord
    # get total cost of the path taken
    path_cost = np.linalg.norm(path[1:]-path[:-1],axis=1).sum()
    print('Cost of optimal path: ', path_cost)
    print('# of iterations to reach goal: ', planner.num_iter)
    print('Time taken: ', time_taken)

    # plot expanded nodes
    expanded_nodes = list(itemgetter(*list(planner.CLOSED))(planner.id_obj_map))
    expanded_node_coords = np.zeros((len(expanded_nodes),3))
    c=0
    for node in expanded_nodes:
        # store coordinates
        expanded_node_coords[c] = node.coord
        c+=1
    # draw the map and optimal path
    fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-')
    # add the expanded node scatter plot
    ax.scatter(expanded_node_coords[:,0][::20],expanded_node_coords[:,1][::20],expanded_node_coords[:,2][::20])
    plt.show()