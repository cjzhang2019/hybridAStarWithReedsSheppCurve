"""
Created on Wed May 13 10:24:24 2020
using Hybrid A* solve planning problem
@author: cjzhang
"""

import numpy as np
import vehicle
import collision_check
import rs_path
import grid_a_star
import math
import queue
import scipy.spatial
import matplotlib.pyplot as plt
import time
import copy

XY_GRID_RESOLUTION = 1.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
GOAL_TYAW_TH = np.deg2rad(5.0)  # [rad]
MOTION_RESOLUTION = 1  # [m] path interporate resolution
N_STEER = 10.0  # number of steer command
EXTEND_AREA = 0.0  # [m] map extend length
SKIP_COLLISION_CHECK = 4  # skip number for collision check

SB_COST = 100.0  # switch back penalty cost
BACK_COST = 5.0  # backward penalty cost
STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
STEER_COST = 1.0  # steer angle change penalty cost
H_COST = 1.0  # Heuristic cost

WB = vehicle.WB  # [m] Wheel base
MAX_STEER = vehicle.MAX_STEER  # [rad] maximum steering angle

class Node(object):

    def __init__(self, xind, yind, yawind, direction, x, y, yaw, directions, steer, cost, pind):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.directions = directions
        self.steer = steer
        #steer input
        self.cost = cost
        self.pind = pind
        # pind::Int64  # parent index

class Path(object):

    # x::Array{Float64} # x position [m]
    # y::Array{Float64} # y position [m]
    # yaw::Array{Float64} # yaw angle [rad]
    # yaw1::Array{Float64} # trailer angle [rad]
    # direction::Array{Bool} # direction forward: true, back false
    # cost::Float64 # cost

    def __init__(self, x, y, yaw, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        self.cost = cost

class Config(object):

    def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw, xw, yw, yaww, xyreso, yawreso):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.xyreso = xyreso
        self.yawreso = yawreso

    def prn_obj(obj):
        print ('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))

def calc_config(ox, oy, xyreso, yawreso):
    #computer capacity

    min_x_m = min(ox) - EXTEND_AREA
    min_y_m = min(oy) - EXTEND_AREA
    max_x_m = max(ox) + EXTEND_AREA
    max_y_m = max(oy) + EXTEND_AREA

    ox.append(min_x_m)
    oy.append(min_y_m)
    ox.append(max_x_m)
    oy.append(max_y_m)

    minx = round(min_x_m/xyreso)
    miny = round(min_y_m/xyreso)
    maxx = round(max_x_m/xyreso)
    maxy = round(max_y_m/xyreso)

    xw = round(maxx - minx)
    yw = round(maxy - miny)

    minyaw = round(- math.pi/yawreso) - 1
    maxyaw = round(math.pi/yawreso)
    yaww = round(maxyaw - minyaw)

    # minyawt = minyaw
    # maxyawt = maxyaw
    # yawtw = yaww

    afterConfig = Config(minx, miny, minyaw, maxx, maxy, maxyaw, xw, yw, yaww, xyreso, yawreso)
    return afterConfig

def calc_index(node, c):
    ind = (node.yawind - c.minyaw)*c.xw*c.yw+(node.yind - c.miny)*c.xw + (node.xind - c.minx)
    # 3D grid
    if ind <= 0:
        print ("Error(calc_index):", ind)

    return ind

def calc_holonomic_with_obstacle_heuristic(Node, ox , oy , xyreso):
    h_dp = grid_a_star.calc_dist_policy(Node.x[-1], Node.y[-1], ox, oy, xyreso, 1.0)
    return h_dp

def calc_cost(n, h_dp, c):

   return (n.cost + 3 * H_COST * h_dp[int(n.xind - c.minx)][int(n.yind - c.miny)])

def calc_motion_inputs():
    up = []
    #0-N_STEER-1
    for i in range(int(N_STEER)-1,-1,-1):
        x = MAX_STEER - i*(MAX_STEER/N_STEER)
        up.append(x)

    u = [0.0] + [i for i in up] + [-i for i in up]
    d = [1.0 for i in range(len(u))] + [-1.0 for i in range(len(u))]
    u = u + u
    #d = 1 move forward d = -1 move backward
    return u, d

def calc_rs_path_cost(rspath):
    #length + forward backward switch + steer + steer switch

    cost = 0.0
    for l in rspath.lengths:
        if l >= 0:  # forward
            cost += l
        else:  # back
            cost += abs(l) * BACK_COST

    # switch back penalty
    for i in range(len(rspath.lengths)-1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:  # switch back
            cost += SB_COST

    # steer penalyty
    for ctype in rspath.ctypes:
        if ctype != "S" : # curve
            cost += STEER_COST * abs(MAX_STEER)

    # steer switch profile
    nctypes = len(rspath.ctypes)
    ulist = [0.0 for i in range(nctypes)]
    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = - MAX_STEER
        elif rspath.ctypes[i] == "L":
            ulist[i] = MAX_STEER

    for i in range(len(rspath.ctypes)-1):
        cost += STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    return cost

def analystic_expantion(n, ngoal, ox, oy, kdtree):
    #here I corrected the mistake

    sx = n.x[-1]
    sy = n.y[-1]
    syaw = n.yaw[-1]

    max_curvature = math.tan(MAX_STEER)/WB
    paths = rs_path.calc_paths(sx, sy, syaw, ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1], max_curvature, step_size = MOTION_RESOLUTION)

    if len(paths) == 0:
        return None

    pathset = {}
    path_id = 0
    for path in paths:
        pathset[path_id] = path
        path_id = path_id + 1

    p_idList = sorted(pathset, key=lambda x: calc_rs_path_cost(pathset[x]))
    for i in p_idList:
        path = pathset[i]
        if collision_check.check_collision(ox, oy, path.x, path.y, path.yaw, kdtree):
            return path # path is ok

    return None

def update_node_with_analystic_expantion(current, ngoal, c, ox, oy, kdtree):

    apath = analystic_expantion(current, ngoal, ox, oy, kdtree)
    if apath != None:
        fx = apath.x[1:]
        fy = apath.y[1:]
        fyaw =  apath.yaw[1:]
        fcost = current.cost + calc_rs_path_cost(apath)
        fpind = calc_index(current, c)

        fd = []
        for d in apath.directions[1:]:
            if d >= 0:
                fd.append(True)
            else:
                fd.append(False)
        fsteer = 0.0
        fpath = Node(current.xind, current.yind, current.yawind, current.direction, fx, fy, fyaw, fd, fsteer, fcost, fpind)
        return True, fpath
    return False, None #no update

def calc_next_node(current, c_id, u, d, c):

    arc_l = XY_GRID_RESOLUTION * 1.5

    nlist = math.ceil(arc_l / MOTION_RESOLUTION) + 1

    xlist, ylist, yawlist = [], [], []

    xlist_0 = current.x[-1] + d * MOTION_RESOLUTION * math.cos(current.yaw[-1])
    ylist_0 = current.y[-1] + d * MOTION_RESOLUTION * math.sin(current.yaw[-1])
    yawlist_0 = rs_path.pi_2_pi(current.yaw[-1] + d * MOTION_RESOLUTION / WB * math.tan(u))
    xlist.append(xlist_0)
    ylist.append(ylist_0)
    yawlist.append(yawlist_0)

    for i in range(1,int(nlist)):
        xlist_i = xlist[i-1] + d * MOTION_RESOLUTION * math.cos(yawlist[i-1])
        ylist_i = ylist[i-1] + d * MOTION_RESOLUTION * math.sin(yawlist[i-1])
        yawlist_i = rs_path.pi_2_pi(yawlist[i-1] + d * MOTION_RESOLUTION / WB * math.tan(u))
        xlist.append(xlist_i)
        ylist.append(ylist_i)
        yawlist.append(yawlist_i)

    xind = round(xlist[-1] / c.xyreso)
    yind = round(ylist[-1] / c.xyreso)
    yawind = round(yawlist[-1] / c.yawreso)

    addedcost = 0.0
    if d > 0:
        direction = True
        addedcost += abs(arc_l)
    else:
        direction = False
        addedcost += abs(arc_l) * BACK_COST

    # swich back penalty
    if direction != current.direction:  # switch back penalty
        addedcost += SB_COST

    # steer penalyty
    addedcost += STEER_COST * abs(u)

    # steer change penalty
    addedcost += STEER_CHANGE_COST * abs(current.steer - u)

    cost = current.cost + addedcost

    directions = [direction for i in range(len(xlist))]
    node = Node(xind, yind, yawind, direction, xlist, ylist, yawlist, directions, u, cost, c_id)

    return node

def verify_index(node, c, ox, oy, kdtree):

    # overflow map
    if (node.xind - c.minx) >= c.xw:
        return False
    elif (node.xind - c.minx) <= 0:
        return False
    if (node.yind - c.miny) >= c.yw:
        return False
    elif (node.yind - c.miny) <= 0:
        return False

    if collision_check.check_collision(ox, oy, node.x, node.y, node.yaw, kdtree) == False:
        return False
    return True #index is ok"

def is_same_grid(node1, node2):

    if node1.xind != node2.xind:
        return False
    if node1.yind != node2.yind:
        return False
    if node1.yawind != node2.yawind:
        return False
    return True

#def get_final_path(closed, ngoal, nstart, c, last_A_star_node_ind):
#    
#    rx1, ry1, ryaw1 = ngoal.x[::-1], ngoal.y[::-1], ngoal.yaw[::-1]
#    
#    direction1 = ngoal.directions[::-1]
#    finalcost1 = ngoal.cost
#    nid = last_A_star_node_ind
#    rx1.extend(closed[last_A_star_node_ind].x[::-1])
#    ry1.extend(closed[last_A_star_node_ind].y[::-1])
#    ryaw1.extend(closed[last_A_star_node_ind].yaw[::-1])
#    direction1.extend(closed[last_A_star_node_ind].directions[::-1])
#    path1 = Path(rx1, ry1, ryaw1, direction1, finalcost1)
#
#    
#    rx2, ry2, ryaw2, direction2, finalcost2 = [], [], [], [], []
#    finalcost2 = closed[nid].cost
#
#    while 1:
#        n = closed[nid]
#        rx2.extend(n.x[::-1])
#        ry2.extend(n.y[::-1])
#        ryaw2.extend(n.yaw[::-1])
#        direction2.extend(n.directions[::-1])
#        nid = n.pind
#        if is_same_grid(n, nstart):
#            break
#    path2 = Path(rx2, ry2, ryaw2, direction2, finalcost2)
#    for i in range(len(rx2)):
#        rx1.append(rx2[i])
#        ry1.append(ry2[i])
#        ryaw1.append(ryaw2[i])
#        direction1.append(direction2[i])
#    finalcost = finalcost1 + finalcost2
#    path = Path(rx1, ry1, ryaw1, direction1, finalcost)
#    
#    
#    return path1, path2, path
def get_final_path(closed, ngoal, nstart, c, last_A_star_node_ind):

    rx, ry, ryaw = ngoal.x[::-1], ngoal.y[::-1], ngoal.yaw[::-1]
    direction = ngoal.directions[::-1]
    nid = last_A_star_node_ind
    finalcost = ngoal.cost
    if(len(rx) == 1):
        rx.append(closed[nid].x[-1])
        ry.append(closed[nid].y[-1])
        ryaw.append(closed[nid].yaw[-1])
        direction.append(closed[nid].directions[-1])
        finalcost = closed[nid].cost
    finalcost1 = finalcost - closed[nid].cost
    rx1 = copy.deepcopy(rx)
    ry1 = copy.deepcopy(ry)
    ryaw1 = copy.deepcopy(ryaw)
    direction1 = copy.deepcopy(direction)
    rx1 = rx1[::-1]
    ry1 = ry1[::-1]
    ryaw1 = ryaw1[::-1]
    direction1 = direction1[::-1]
    path1 = Path(rx1, ry1, ryaw1, direction1, finalcost1)

    while 1:
        n = closed[nid]
        rx.extend(n.x[::-1])
        ry.extend(n.y[::-1])
        ryaw.extend(n.yaw[::-1])
        direction.extend(n.directions[::-1])
        nid = n.pind
        if is_same_grid(n, nstart):
            break
    rx = rx[::-1]
    ry = ry[::-1]
    ryaw = ryaw[::-1]
    direction = direction[::-1]

    # adjuct first direction
    direction[0] = direction[1]

    path = Path(rx, ry, ryaw, direction, finalcost)
    

    return path1, path

class KDTree:
    """
    Nearest neighbor search class with KDTree
    Dimension is two
    """

    def __init__(self, data):
        # store kd-tree
        self.tree = scipy.spatial.cKDTree(data)

    def search(self, inp, k=1):
        """
        k=1 means to query the nearest neighbours and return squeezed result
        inp: input data
        """

        if len(inp.shape) >= 2:  # multi input 
            index = []
            dist = []
            for i in inp.T:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)

            return index, dist
        else:
            dist, index = self.tree.query(inp, k=k)
            return index, dist

    def search_in_distance(self, inp, r):
        """
        find points within a distance r
        """
        index = self.tree.query_ball_point(inp, r)
        return index

def calc_hybrid_astar_path(sx , sy , syaw , gx , gy , gyaw ,  ox , oy , xyreso , yawreso):

    # sx: start x position[m]
    # sy: start y position[m]
    # gx: goal x position[m]
    # gy: goal y position[m]
    # ox: x position list of Obstacles[m]
    # oy: y position list of Obstacles[m]
    # xyreso: grid resolution[m]
    # yawreso: yaw angle resolution[rad]

    syaw0 = rs_path.pi_2_pi(syaw)
    gyaw0 = rs_path.pi_2_pi(gyaw)
    #keep -pi-pi
    global tox,toy
    ox, oy = ox[:], oy[:]
    tox, toy = ox[:], oy[:]
    kdtree = KDTree(np.vstack((tox, toy)).T)
    #use kdtree to represent obstacles logN < N

    c = calc_config(ox, oy, xyreso, yawreso)
    nstart = Node(round(sx / xyreso), round(sy / xyreso), round(syaw0 / yawreso), True, [sx], [sy], [syaw0], [True], 0.0, 0.0, -1)
    ngoal = Node(round(gx/xyreso), round(gy/xyreso), round(gyaw0/yawreso), True, [gx], [gy], [gyaw0], [True], 0.0, 0.0, -1)
    h_dp = calc_holonomic_with_obstacle_heuristic(ngoal, ox, oy, xyreso)
    #cost from the goal to each point index
    openset, closedset = {},{}
    fnode = ngoal
    openset[calc_index(nstart, c)] = nstart

    u, d = calc_motion_inputs()
    nmotion = len(u)

    if collision_check.check_collision(ox, oy, [sx], [sy], [syaw0], kdtree) == False:
        return []
    if collision_check.check_collision(ox, oy, [gx], [gy], [gyaw0], kdtree) == False:
        return []
    times = 0
    last_A_star_node_ind = 0
    while 1:
#        if times > 1000:
#            return [],[]
        if len(openset) == 0:
            print ("Error: Cannot find path, No open set")
            return []

        c_id = min(openset, key=lambda o: calc_cost(openset[o], h_dp, c))

        current = openset[c_id]

        # move current node from open to closed
        del openset[c_id]
        closedset[c_id] = current
        plt.plot(current.x[::-1],current.y[::-1],"rx")
        
        if ((current.x[-1] - gx) ** 2 + (current.y[-1] - gy) ** 2  < 1 and abs(current.yaw[-1] - gyaw) < 0.35):
            last_A_star_node_ind = c_id
            break

        isupdated, fpath = update_node_with_analystic_expantion(current, ngoal, c, ox, oy, kdtree)
        if isupdated:  # found
            last_A_star_node_ind = c_id
            fnode = fpath
            break


        for i in range(nmotion):
            node = calc_next_node(current, c_id, u[i], d[i], c)

            if verify_index(node, c, ox, oy, kdtree) == False:
                continue

            node_ind = calc_index(node, c)

            # If it is already in the closed set, skip it
            if node_ind in closedset:
                continue

            if node_ind not in openset:
                openset[node_ind] = node
                
            else:
                if calc_cost(openset[node_ind], h_dp, c) > calc_cost(node, h_dp, c):
                    # If so, update the node to have a new parent
                    openset[node_ind] = node
        times = times + 1
    path1, path = get_final_path(closedset, fnode, nstart, c, last_A_star_node_ind)
    print(times)
    plt.savefig("searchPoint2d.png", format='png', bbox_inches='tight')
    plt.show()

    return path1, path

def show_animation(path, oox, ooy):
    x = path.x
    y = path.y
    yaw = path.yaw
    direction = path.direction
    steer = 0.0
    plt.figure(1)
    for ii in range(0,len(x),1):
        plt.cla()
        plt.plot(oox, ooy, ".k")
        plt.plot(x, y, "-r", label="Hybrid A* path")

        if ii < len(x)-1:
            k = (yaw[ii+1] - yaw[ii])/MOTION_RESOLUTION
            if direction[ii] == False:
                k *= -1
            steer = math.atan2(collision_check.WB*k, 1.0)
        else:
            steer = 0.0
        vehicle.plot_trailer(x[ii], y[ii], yaw[ii], steer)
        plt.grid(True)
        plt.axis("equal")
        plt.pause(0.01)

def main():
    start = time.time()
    ox = []
    oy = []
    obstacle = []
        
    obstacle.append((-5,-10))
    obstacle.append((-5,10))
    obstacle.append((35,10))
    obstacle.append((35,-10))
    SX = 0  # [m]
    SY = 0  # [m]
    SYAW = np.deg2rad(0)
    
    
#    GX = 0  # [m]
#    GY = 4 # [m]
#    GYAW = np.deg2rad(180.0)
#    for i in range(20):
#        obstacle.append((8,10-i))
#    for i in range(40):
#        obstacle.append((-5+i,10))
#    for i in range(40):
#        obstacle.append((-5+i,-10))
    #case1
    
#    GX = 0  # [m]
#    GY = 4 # [m]
#    GYAW = np.deg2rad(180.0)
#    for i in range(8):
#        obstacle.append((-5,6-i))
#    for i in range(8):
#        obstacle.append((4,6-i))
#    for i in range(10):
#        obstacle.append((-5+i,6))
#    for i in range(10):
#        obstacle.append((-5+i,-2))
    #case2
    
#    GX = 30  # [m]
#    GY = 3 # [m]
#    GYAW = np.deg2rad(0.0)
#    for i in range(40):
#        obstacle.append((-5+i,10))
#    for i in range(40):
#        obstacle.append((-5+i,-10))
    #case3
    
#    GX = 30  # [m]
#    GY = -5 # [m]
#    GYAW = np.deg2rad(0.0)
#    for i in range(4):
#        obstacle.append((2 * i-5,-2))
#    obstacle.append((3,3))
#    obstacle.append((3,-2))
#    obstacle.append((5,3))
#    obstacle.append((5,-2))
#    obstacle.append((7,4))
#    obstacle.append((7,-1))
#    obstacle.append((9,5))
#    obstacle.append((9,0))
#    obstacle.append((11,5))
#    obstacle.append((11,0))
#    obstacle.append((13,5))
#    obstacle.append((13,0))
#    obstacle.append((15,5))
#    obstacle.append((15,0))
#    obstacle.append((17,4))
#    obstacle.append((17,-1))
#    obstacle.append((19,3))
#    obstacle.append((19,-2))
#    obstacle.append((21,2))
#    obstacle.append((21,-3))
#    obstacle.append((23,1))
#    obstacle.append((23,-4))
#    obstacle.append((25,0))
#    obstacle.append((25,-5))
#    obstacle.append((27,-1))
#    obstacle.append((27,-6))
#    obstacle.append((29,-2))
#    obstacle.append((29,-7))
    #case4
    
#    GX = 25.0
#    GY = 0.0
#    GYAW = np.deg2rad(0.0)
#    for i in range(30):
#        obstacle.append((i-5,5))
#    for i in range(30):
#        obstacle.append((i-5,-3))
#    for i in range(5):
#        obstacle.append((i+8,1.5))
#    for i in range(5):
#        obstacle.append((i+8,-1.5))
#    for i in range(7):
#        obstacle.append((8,1.5-i*0.5))
#    for i in range(7):
#        obstacle.append((12,1.5-i*0.5))
    #case5

#    GX = -1.0
#    GY = -10
#    GYAW = np.deg2rad(180.0)
#    for i in range(10):
#        obstacle.append((i,3))
#    for i in range(5):
#        obstacle.append((10+i,3-i))
#    for i in range(6):
#        obstacle.append((14,-2-i))
#    for i in range(13):
#        obstacle.append((9-i,-13))
#    for i in range(5):
#        obstacle.append((14-i,-8-i))
#
#    for i in range(15):
#        obstacle.append((i-5,-8))
#    for i in range(15):
#        obstacle.append((i-5,-2))
#    for i in range(5):
#        obstacle.append((9,-3-i))
    #case6
    
    GX = 27
    GY = -8
    GYAW = np.deg2rad(90.0)
    for i in range(7):
        obstacle.append((25, i-10))
    for i in range(7):
        obstacle.append((29, i-10))
    for i in range(7):
        obstacle.append((10, i-10))
    for i in range(7):
        obstacle.append((14, i-10))

    for i in range(6):
        obstacle.append((6,i-3))
    for i in range(6):
        obstacle.append((13,i+5))
    for i in range(6):
        obstacle.append((20,i-5))
        
    for i in range(40):
        obstacle.append((-5+i,10))
    for i in range(40):
        obstacle.append((-5+i,-10))
    #case7
    
    for (x,y) in obstacle:
        ox.append(x)
        oy.append(y)

    
    PATH1, PATH = calc_hybrid_astar_path(SX , SY , SYAW , GX , GY , GYAW ,  ox , oy , XY_GRID_RESOLUTION , YAW_GRID_RESOLUTION)
    if PATH1 == [] and PATH == []:
        print ('Error: Cannot find path, overtime')
        return
#    vehicle.plot_trailer(SX, SY, SYAW, 0)
#    show_animation(PATH, ox, oy)
    plt.plot(ox, oy, ".k")
    plt.plot(SX, SY, "xr")
    plt.plot(GX, GY, "xb")
    plt.grid(True)
    
    plt.axis("equal")
    rx1, ry1 = PATH1.x, PATH1.y
    rx, ry = PATH.x, PATH.y

    font1 = {'family' : 'Times New Roman','weight' : 'normal', 'size'   : 20}
    plt.plot(rx, ry, "-r")
    plt.plot(rx1, ry1, "-b")
    plt.xlabel('x/(m)',font1)
    
    plt.ylabel('y/(m)',font1)
    plt.legend(['Obstacles', 'VehiclePose', 'Destination', 'Path', 'Path using Reeds-Shepp curve'],prop=font1, bbox_to_anchor=(1.05, 0), loc=3, borderaxespad = 0)
    
    plt.title('Path Planning Using Hybrid A Star\n(Reeds-Shepp curve)', font1)
    plt.savefig('Path Planning Using Hybrid A Star.png', format='png', bbox_inches='tight')
    plt.show()


    elapsed = (time.time() - start)
    
    print("Time used:",elapsed)
    
    
    
if __name__ == '__main__':
    main()