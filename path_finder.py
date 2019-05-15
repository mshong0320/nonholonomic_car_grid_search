from waypoint import Waypoint
import matplotlib as plt
import numpy as np
from math import sqrt
import sys
import Queue


# for part(iii) of the challenge:
# part i) is done in the first where I have put the first '###########' and part b) 
# comes along on the next '#############'.
# For part a) I implemented the greedy BFS algorithm, which is not complete, not optimal,
# and has a time complexity of O(b^n) where b is branching factor, which in this case is max 6 per element in the grid, and n here 
# depends on the queues. For part ii) I used A* algorithm, which is also O(b^n).

class PathFinder(object):

    def get_path(self, grid, start_wp, end_wp):
        """Returns a list of Waypoints from the start Waypoint to the end Waypoint.
        :param grid: Grid is a 2D numpy ndarray of boolean values. grid[x, y] == True if the cell contains an obstacle.
        The grid dimensions are exposed via grid.shape
        :param start_wp: The Waypoint that the path should start from.
        :param end_wp: The Waypoint that the path should end on.
        :return: The path from the start waypoint to the end waypoint that follows the movement model without going
        off the grid or intersecting an obstacle.
        :rtype: A list of Waypoints.

        More documentation at
        https://docs.google.com/document/d/1b30L2LeKyMjO5rBeCui38j_HSUYgEGWXrwSRjB7AnYs/edit?usp=sharing
        """
        #initializing waypoints list

        row = grid.shape[0]
        col = grid.shape[1]

        startx = start_wp.x
        starty = start_wp.y
        goalx = end_wp.x
        goaly = end_wp.y
        goalo = end_wp.orientation

        path = [] 
        openset = [] # this is the openList
        visited = [] # this is the closedList

        Frontier = Queue.PriorityQueue()
        parent = {}
        cell_heuristic = {} # this is to keep the heuristics 
        heuristic = np.array(grid)
        distance = np.array(grid)
        dist_away_from_start = {}
        goal_reached = False
        # directions = [[0, 1],  # drive forwards
        #               [1, 1],  # drive right front
        #               [-1, 1],  # go left front
        #               [0, -1],   # drive backwards
        #               [1, -1],   # drive right backwards
        #               [-1, -1]]  # drive left backwards

        def euclidean_heuristics(x,y, gx, gy):
            dist = sqrt((gx-x)**2+(gy-y)**2)
            return dist

        def how_far_away(x, y, sx, sy):
            count = abs(sx-x)+abs(sy-y)
            return count

        # given the x, y, and orientation of the robot at the current cell find all its neighbors and their config
        def neighbors(ix, iy, io, Grid): 
            neighbors = []
            grid_x = Grid.shape[0]-1
            grid_y = Grid.shape[1]-1
            if ix == 0 and iy == 0:
                if io == 0 or io == 2:
                    neighbors = [(ix, iy+1, io), (ix+1, iy+1, 1)]
                elif io == 1:
                    neighbors = [(ix+1,iy, io), (ix+1, iy+1, 0)]
                elif io == 3:
                    neighbors = [(ix+1,iy, io), (ix+1, iy+1, 2)]
            elif ix == 0 and iy > 0 and iy < grid_y:
                if io == 0:
                    neighbors = [(ix, iy+1, io), (ix, iy-1, io), (ix+1, iy+1, 1), (ix+1, iy-1, 3)]
                elif io == 2:
                    neighbors = [(ix, iy+1, io), (ix, iy-1, io), (ix+1, iy+1, 3), (ix+1, iy-1, 1)]
                elif io == 1:
                    neighbors = [(ix+1, iy+1,0),(ix+1, iy-1, 2),(ix+1, iy, 1)]
                elif io == 3:
                    neighbors = [(ix+1, iy+1,2),(ix+1, iy-1, 0),(ix+1, iy, 3)]
            elif (grid_x > ix > 0) and (grid_y > iy > 0):
                if io == 0:
                    neighbors = [(ix, iy+1,io), (ix,iy-1,io), (ix+1, iy+1, 1), (ix-1, iy+1, 3), (ix+1, iy-1,3), (ix-1, iy-1,1)]
                elif io == 2:
                    neighbors = [(ix, iy+1,io), (ix,iy-1,io), (ix+1, iy+1, 3), (ix-1, iy+1, 1), (ix+1, iy-1,1), (ix-1, iy-1,3)]
                elif io == 1:
                    neighbors = [(ix+1, iy, io), (ix+1, iy-1, 2), (ix+1,iy+1, 0), (ix-1, iy, io), (ix-1, iy+1, 2), (ix-1, iy-1, 0)]
                elif io == 3:
                    neighbors = [(ix+1, iy, io), (ix+1, iy-1, 0), (ix+1,iy+1, 2), (ix-1, iy, io), (ix-1, iy+1, 0), (ix-1, iy-1, 2)]
            elif ix == grid_x and (0 < iy < grid_y):
                if io == 0:
                    neighbors = [(ix, iy+1, io), (ix-1, iy+1, 3), (ix-1, iy-1, 1), (ix, iy-1, 0)]
                elif io == 1:
                    neighbors = [(ix-1, iy+1, 2), (ix-1, iy-1, 0), (ix-1, iy, io)]
                elif io == 2:
                    neighbors = [(ix, iy+1, io), (ix, iy-1, io), (ix-1, iy+1, 1), (ix-1, iy-1, 3)]
                elif io == 3:
                    neighbors = [(ix-1, iy, 0), (ix-1, iy+1, 0), (ix-1, iy-1, 2)]
            elif iy == grid_y and ix == grid_x:
                if io == 0:
                    neighbors = [(ix, iy-1, 0), (ix-1, iy-1, 1)]
                elif io == 1:
                    neighbors = [(ix-1, iy, 1), (ix-1, iy-1, 0)]
                elif io == 2:
                    neighbors = [(ix, iy-1, 2), (ix-1, iy-1, 3)]
                elif io == 3:
                    neighbors = [(ix-1, iy, 3), (ix-1, iy-1, 2)]
            elif iy == grid_y and ix < grid_x:
                if io==0:
                    neighbors = [(ix, iy-1, io), (ix-1, iy-1, 1), (ix+1, iy-1, 3)]
                elif io==1:
                    neighbors = [(ix+1, iy, io), (ix-1, iy, io), (ix+1, iy-1, 2), (ix-1, iy-1, 0)]
                elif io == 2:
                    neighbors = [(ix, iy-1, io), (ix-1, iy-1, 3), (ix+1, iy-1, 1)]
                elif io == 3:
                    neighbors = [(ix-1, iy, io), (ix+1, iy, io), (ix-1, iy-1, 2), (ix+1, iy-1, 0)]
            return neighbors

         # sanity check for start_wp and end_wp within our grid
        if (0 <= start_wp.x <= row) & (0 <= start_wp.y <= col) & (0 <= end_wp.x <= row) & (0 <= end_wp.y <= col):
            okay_to_start = True
            print("is it okay_to_start? :", okay_to_start)

        # # setting up temp distances of each cell, and the obstacles as 1000
        for i in range(row):
            for j in range(col):
                if grid[i,j] != True:
                    distance[i,j] = how_far_away(i,j,startx, starty)
                else:
                    distance[i,j] = 1000

        openset.append(start_wp)
        distance[start_wp.x, start_wp.y] = 0

        #######################################################################
        # for challenge (i) with no obstacles: greedy search algorithm is used here
        current = start_wp
        if np.sum(grid) == 0:
            path.append(start_wp)
            while end_wp not in path:
                if len(openset) != 0:
                    visited.append(current)
                    openset.remove(current)

                curr_orient = current.orientation
                # finding the children of the current node
                current_neighbors = neighbors(current.x, current.y, curr_orient, grid) # array of neighbors of current cell
                
                # go through all the neighboring cells to add them in the openset
                for i in range(len(current_neighbors)):

                    ix = current_neighbors[i][0]
                    iy = current_neighbors[i][1]
                    
                    # to check if the neighboring cell is not an obstacle
                    if grid[ix,iy] != True: 
                        # if it is not an obstacle
                        iix = current_neighbors[i][0]
                        iiy = current_neighbors[i][1]
                        iio = current_neighbors[i][2]

                        nextterm = Waypoint(iix, iiy, iio)
                        if nextterm not in openset:
                            openset.append(nextterm)

                            # for greedy search heuristic:
                            cell_heuristic[nextterm] = euclidean_heuristics(ix,iy,goalx,goaly)
                    else:
                        continue

                # at each current cell, find the neighbor with the lowest updated heuristics
                neigh_w_smallest_cost = min(cell_heuristic, key=cell_heuristic.get)
                nextpoint_x = neigh_w_smallest_cost.x
                nextpoint_y = neigh_w_smallest_cost.y
                nextpoint_o = neigh_w_smallest_cost.orientation

                nextpoint = Waypoint(nextpoint_x, nextpoint_y, nextpoint_o)
                path.append(nextpoint)
                current = nextpoint

                # correcting orientation of the car at goal
                if current.x == goalx and current.y == goaly and current.orientation != goalo:
                    current_ori = current.orientation
                    current_x = current.x
                    current_y = current.y

                    if goalo == 3 and current_ori == 1:
                        orientation_fix = [Waypoint(current_x+1, current_y+1, 0), Waypoint(current_x+1, current_y, 0), Waypoint(current_x+1, current_y-1, 0), Waypoint(current_x, current_y, 3)]
                    elif goalo == 1 and current_ori == 2:
                        orientation_fix = [Waypoint(current_x, current_y-1, 2), Waypoint(current_x-1, current_y, 1), Waypoint(current_x, current_y, 1)]
                    elif goalo == 0 and current_ori == 2:
                        orientation_fix = [Waypoint(current_x-1, current_y+1, 1), Waypoint(current_x, current_y+2, 0), Waypoint(current_x, current_y+1, 0), Waypoint(current_x, current_y, 0)]
                    elif goalo == 2 and current_ori == 0:
                        orientation_fix = [Waypoint(current_x-1, current_y-1, 1), Waypoint(current_x, current_y-2, 2), Waypoint(current_x, current_y-1, 2), Waypoint(current_x, current_y, 2)]
                    elif goalo == 1 and current_ori ==3:
                        orientation_fix = [Waypoint(current_x+1, current_y+1, 2), Waypoint(current_x+1, current_y, 2), Waypoint(current_x+1, current_y-1, 2), Waypoint(current_x, current_y, 1)]
                    elif goalo == 3 and current_ori == 2:
                        orientation_fix = [Waypoint(current_x, current_y-1, 2), Waypoint(current_x+1, current_y, 3), Waypoint(current_x, current_y, 3)]
                    elif goalo == 2 and current_ori == 1:
                        orientation_fix = [Waypoint(current_x+1, current_y, 1), Waypoint(current_x, current_y+1, 2), Waypoint(current_x, current_y, 2)]
                    elif goalo == 0 and current_ori == 1:
                        orientation_fix = [Waypoint(current_x+1, current_y, 0), Waypoint(current_x, current_y-1, 0), Waypoint(current_x, current_y, 0)]
                    elif goalo == 2 and current_ori == 3:
                        orientation_fix = [Waypoint(current_x-1, current_y, 3), Waypoint(current_x, current_y+1, 2), Waypoint(current_x, current_y, 2)]
                    elif goalo == 0 and current_ori == 3:
                        orientation_fix = [Waypoint(current_x-1, current_y, 3), Waypoint(current_x, current_y-1, 0), Waypoint(current_x, current_y, 0)]
                    elif goalo == 1 and current_ori == 0:
                        orientation_fix = [Waypoint(current_x, current_y-1, 0), Waypoint(current_x+1, current_y, 1), Waypoint(current_x, current_y, 1)]
                    elif goalo == 3 and current_ori == 0:
                        orientation_fix = [Waypoint(current_x, current_y-1, 0), Waypoint(current_x-1, current_y, 3), Waypoint(current_x, current_y, 3)]
                    
                    for item in range(len(orientation_fix)):
                        path.append(orientation_fix[item])
                else:
                    continue
            
            print path
            print("length of the obtained path: ", path)
            return path
        
        ##############################################################################
        # challenge(ii) begins here: added more efficiency by using A* 
        else:
            dist_away_from_start[start_wp] = 0
            parent[start_wp] = None
            Frontier.put(start_wp, 0)
            
            # I'm going to switch to A* search from here on with a euclidean heuristic function 
            while not Frontier.empty():
                current = Frontier.get()

                if current == end_wp:
                    break

                curr_orient = current.orientation

                # array of neighbors of current cell
                current_neighbors = neighbors(current.x, current.y, curr_orient, grid) 
                obstacle_free_neighbors = []

                for i in range(len(current_neighbors)):
                    ix = current_neighbors[i][0]
                    iy = current_neighbors[i][1]

                    if grid[ix, iy] != True:
                        new_cost = dist_away_from_start[current]+distance[ix, iy]
                        iix = current_neighbors[i][0]
                        iiy = current_neighbors[i][1]
                        iio = current_neighbors[i][2]

                        Next = Waypoint(iix, iiy, iio)

                        if Next not in dist_away_from_start or new_cost < dist_away_from_start[Next]:
                            dist_away_from_start[Next] = new_cost
                            heuristic = euclidean_heuristics(iix,iiy,goalx,goaly)
                            priority = new_cost + heuristic
                            Frontier.put(Next, priority)
                            parent[Next] = current

            print("The goal has been found! Now lets build the path")

            while current != start_wp:
                path.append(current)
                current = parent[current]

            path.append(start_wp)
            path.reverse()
            
            # correcting orientation of the car at goal
            if current.x == goalx and current.y == goaly and current.orientation != goalo:
                current_ori = current.orientation
                current_x = current.x
                current_y = current.y
                if goalo == 3 and current_ori == 1:
                    orientation_fix = [Waypoint(current_x+1, current_y+1, 0), Waypoint(current_x+1, current_y, 0), Waypoint(current_x+1, current_y-1, 0), Waypoint(current_x, current_y, 3)]
                elif goalo == 1 and current_ori == 2:
                    orientation_fix = [Waypoint(current_x, current_y-1, 2), Waypoint(current_x-1, current_y, 1), Waypoint(current_x, current_y, 1)]
                elif goalo == 0 and current_ori == 2:
                    orientation_fix = [Waypoint(current_x-1, current_y+1, 1), Waypoint(current_x, current_y+2, 0), Waypoint(current_x, current_y+1, 0), Waypoint(current_x, current_y, 0)]
                elif goalo == 2 and current_ori == 0:
                    orientation_fix = [Waypoint(current_x-1, current_y-1, 1), Waypoint(current_x, current_y-2, 2), Waypoint(current_x, current_y-1, 2), Waypoint(current_x, current_y, 2)]
                elif goalo == 1 and current_ori ==3:
                    orientation_fix = [Waypoint(current_x+1, current_y+1, 2), Waypoint(current_x+1, current_y, 2), Waypoint(current_x+1, current_y-1, 2), Waypoint(current_x, current_y, 1)]
                elif goalo == 3 and current_ori == 2:
                    orientation_fix = [Waypoint(current_x, current_y-1, 2), Waypoint(current_x+1, current_y, 3), Waypoint(current_x, current_y, 3)]
                elif goalo == 2 and current_ori == 1:
                    orientation_fix = [Waypoint(current_x+1, current_y, 1), Waypoint(current_x, current_y+1, 2), Waypoint(current_x, current_y, 2)]
                elif goalo == 0 and current_ori == 1:
                    orientation_fix = [Waypoint(current_x+1, current_y, 0), Waypoint(current_x, current_y-1, 0), Waypoint(current_x, current_y, 0)]
                elif goalo == 2 and current_ori == 3:
                    orientation_fix = [Waypoint(current_x-1, current_y, 3), Waypoint(current_x, current_y+1, 2), Waypoint(current_x, current_y, 2)]
                elif goalo == 0 and current_ori == 3:
                    orientation_fix = [Waypoint(current_x-1, current_y, 3), Waypoint(current_x, current_y-1, 0), Waypoint(current_x, current_y, 0)]
                elif goalo == 1 and current_ori == 0:
                    orientation_fix = [Waypoint(current_x, current_y-1, 0), Waypoint(current_x+1, current_y, 1), Waypoint(current_x, current_y, 1)]
                elif goalo == 3 and current_ori == 0:
                    orientation_fix = [Waypoint(current_x, current_y-1, 0), Waypoint(current_x-1, current_y, 3), Waypoint(current_x, current_y, 3)]
                    
                for item in range(len(orientation_fix)):
                    path.append(orientation_fix[item])
            else:
                pass

            print path
            print("length of the obtained path: ", len(path))
            return path