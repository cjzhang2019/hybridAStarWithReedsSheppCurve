"""
Created on Sun May 24 16:14:23 2020
for collision check
@author: cjzhang
"""
import math
import matplotlib.pyplot as plt
import vehicle

WBUBBLE_DIST = (vehicle.LB + vehicle.LF) / 2.0 - vehicle.LB #[m] distance from rear and the center of whole bubble
WBUBBLE_R = (vehicle.LB + vehicle.LF) / 2.0 #[m] whole bubble radius
VRX = [vehicle.LF, vehicle.LF, -vehicle.LB, -vehicle.LB, vehicle.LF]
VRY = [-vehicle.W/ 2.0, vehicle.W / 2.0,vehicle.W / 2.0, -vehicle.W / 2.0, -vehicle.W / 2.0]

def rect_check(ix, iy, iyaw, ox, oy, vrx, vry):

    for (iox, ioy) in zip(ox, oy):
        tx = iox - ix
        ty = ioy - iy
        lx = (math.cos(iyaw) * tx + math.sin(iyaw) * ty)
        ly = (math.cos(iyaw) * ty - math.sin(iyaw) * tx)

        sumangle = 0.0
        for i in range(len(vrx)-1):
            x1 = vrx[i] - lx
            y1 = vry[i] - ly
            x2 = vrx[i+1] - lx
            y2 = vry[i+1] - ly
            d1 = math.hypot(x1,y1)
            d2 = math.hypot(x2,y2)
            tmp = (x1 * x2 + y1 * y2) / (d1 * d2)
            sumangle += math.acos(tmp)

        if abs(sumangle - 2 * math.pi) < 0.001:
            return False #collision
    return True #OK


def check_collision(ox, oy, x, y, yaw, kdtree):


    for (ix, iy, iyaw) in zip(x, y, yaw):

        cx = ix + WBUBBLE_DIST * math.cos(iyaw)
        cy = iy + WBUBBLE_DIST * math.sin(iyaw)

        # Whole bubble check
        ids = kdtree.search_in_distance([cx, cy], WBUBBLE_R)
        if len(ids) == 0:
            continue
        vrx = VRX
        vry = VRY
        temp_ox, temp_oy = [],[]
        for i in ids:
            temp_ox.append(ox[i])
            temp_oy.append(oy[i])

        if rect_check(ix, iy, iyaw, temp_ox, temp_oy, vrx, vry) == False:
            return False #collision

    return True  # OK




