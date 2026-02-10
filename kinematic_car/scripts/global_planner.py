#!/usr/bin/env python3
import rospy
import numpy as np
import math
import heapq as hq
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry

try:
    from obstacle_detector.msg import Polytope2DArray
except ImportError:
    from kinematic_car.msg import Polytope2DArray

# --- 1. OPTIMIZED GRID MAP ---
class GridMap:
    def __init__(self, bounds, cell_size):
        self.bounds = bounds
        self.cell_size = cell_size
        self.Nx = math.ceil((bounds[1][0] - bounds[0][0]) / cell_size)
        self.Ny = math.ceil((bounds[1][1] - bounds[0][1]) / cell_size)
        
        # Binary Occupancy Grid (False = Free, True = Occupied)
        # This makes lookups O(1) instead of doing math
        self.occupancy = np.zeros((self.Nx, self.Ny), dtype=bool)

    def get_index(self, pos):
        i = math.floor((pos[0] - self.bounds[0][0]) / self.cell_size)
        j = math.floor((pos[1] - self.bounds[0][1]) / self.cell_size)
        if 0 <= i < self.Nx and 0 <= j < self.Ny:
            return i, j
        return None, None

    def get_pos(self, i, j):
        x = self.bounds[0][0] + (i + 0.5) * self.cell_size
        y = self.bounds[0][1] + (j + 0.5) * self.cell_size
        return np.array([x, y])

    def is_occupied(self, i, j):
        if 0 <= i < self.Nx and 0 <= j < self.Ny:
            return self.occupancy[i, j]
        return True # Treat out of bounds as occupied

# --- 2. OBSTACLE MATH ---
class PolytopeRegion:
    def __init__(self, vertices):
        self.A = []
        self.b = []
        # Create Half-spaces Ax <= b
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]]) # Normal (-y, x)
            normal = normal / np.linalg.norm(normal)
            self.A.append(normal)
            self.b.append(np.dot(normal, p1))
        self.A = np.array(self.A)
        self.b = np.array(self.b)

# --- 3. GRAPH SEARCH (A* + LoS) ---
class Node:
    def __init__(self, idx, parent=None, g=math.inf, h=math.inf):
        self.idx = idx # (i, j) tuple
        self.parent = parent
        self.g = g
        self.f = g + h
        self.pos = None # Filled on demand

    def __lt__(self, other):
        return self.f < other.f

class GraphSearch:
    def __init__(self, grid_map):
        self.grid = grid_map

    def a_star(self, start_pos, goal_pos):
        start_idx = self.grid.get_index(start_pos)
        goal_idx = self.grid.get_index(goal_pos)

        if not start_idx or not goal_idx:
            rospy.logwarn("Start or Goal out of bounds")
            return []

        if self.grid.is_occupied(*start_idx) or self.grid.is_occupied(*goal_idx):
            rospy.logwarn("Start or Goal is inside an obstacle")
            return []

        open_set = []
        
        # Heuristic: Euclidean distance
        def heuristic(idx_a, idx_b):
            pos_a = self.grid.get_pos(*idx_a)
            pos_b = self.grid.get_pos(*idx_b)
            return np.linalg.norm(pos_a - pos_b)

        start_node = Node(start_idx, g=0, h=heuristic(start_idx, goal_idx))
        hq.heappush(open_set, (start_node.f, start_node))
        
        came_from = {} # Keep track of visited nodes: idx -> Node
        came_from[start_idx] = start_node

        while open_set:
            current = hq.heappop(open_set)[1]

            if current.idx == goal_idx:
                return self.reconstruct_path(current)

            # 8-connected neighbors
            i, j = current.idx
            neighbors = [
                (i+1, j), (i-1, j), (i, j+1), (i, j-1),
                (i+1, j+1), (i-1, j-1), (i+1, j-1), (i-1, j+1)
            ]

            for ni, nj in neighbors:
                # 1. Check Bounds & Collision (O(1) lookup)
                if self.grid.is_occupied(ni, nj):
                    continue

                # 2. Calc Cost
                step_cost = math.sqrt((ni-i)**2 + (nj-j)**2) * self.grid.cell_size
                tentative_g = current.g + step_cost
                
                neighbor_idx = (ni, nj)
                
                if neighbor_idx not in came_from or tentative_g < came_from[neighbor_idx].g:
                    new_node = Node(neighbor_idx, parent=current, g=tentative_g, h=heuristic(neighbor_idx, goal_idx))
                    came_from[neighbor_idx] = new_node
                    hq.heappush(open_set, (new_node.f, new_node))

        return []

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(self.grid.get_pos(*node.idx))
            node = node.parent
        
        # Reverse to get Start -> Goal
        raw_path = path[::-1]
        
        # Run Line-of-Sight Smoothing (AstarLoS)
        return self.reduce_path(raw_path)

    def reduce_path(self, path):
        """
        Greedy String Pulling:
        Keep connecting current node to the furthest visible node in the list.
        """
        if len(path) < 3: return path
        
        smooth_path = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Check backwards from the end to find the first visible point
            for check_idx in range(len(path)-1, current_idx, -1):
                if self.line_of_sight(path[current_idx], path[check_idx]):
                    smooth_path.append(path[check_idx])
                    current_idx = check_idx
                    break
        return smooth_path

    def line_of_sight(self, p1, p2):
        """
        Bresenham-like check or Raycast on the grid
        """
        dist = np.linalg.norm(p2 - p1)
        if dist < self.grid.cell_size: return True

        steps = math.ceil(dist / (self.grid.cell_size / 2.0))
        for k in range(1, steps):
            t = k / steps
            pt = p1 + (p2 - p1) * t
            idx = self.grid.get_index(pt)
            if idx and self.grid.is_occupied(*idx):
                return False
        return True

# --- 4. ROS NODE ---
class GlobalPlannerNode:
    def __init__(self):
        rospy.init_node('global_planner')

        self.frame_id = rospy.get_param('~frame_id', 'map')
        self.margin = rospy.get_param('~margin', 0.08)
        self.cell_size = rospy.get_param('~cell_size', 0.05) # Larger cell size = Faster
        
        # Bounds slightly larger than maze
        self.bounds = ((-1.0, -1.0), (3.0, 2.0))
        
        self.grid_map = GridMap(self.bounds, self.cell_size)
        self.polytopes = []
        self.current_pos = None
        self.current_goal = None

        self.sub_odom = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_cb)
        self.sub_goal = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)
        self.sub_obs = rospy.Subscriber('/obstacles', Polytope2DArray, self.obs_cb)
        self.pub_path = rospy.Publisher('/global_path', Path, queue_size=1)

    def odom_cb(self, msg):
        self.current_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

    def goal_cb(self, msg):
        self.current_goal = np.array([msg.pose.position.x, msg.pose.position.y])
        rospy.loginfo("New Goal. Planning...")
        self.plan()

    def obs_cb(self, msg):
        # 1. Parse Obstacles
        new_polytopes = []
        for poly_msg in msg.obstacles:
            verts = [np.array([p.x, p.y]) for p in poly_msg.vertices]
            if len(verts) >= 3:
                new_polytopes.append(PolytopeRegion(np.array(verts)))
        
        self.polytopes = new_polytopes
        
        # 2. UPDATE OCCUPANCY GRID (The performance fix!)
        rospy.loginfo("Updating Occupancy Grid...")
        for i in range(self.grid_map.Nx):
            for j in range(self.grid_map.Ny):
                pt = self.grid_map.get_pos(i, j)
                occupied = False
                for poly in self.polytopes:
                    # Check collision with margin
                    margins = self.margin * np.linalg.norm(poly.A, axis=1)
                    if np.all((poly.A @ pt - poly.b - margins) <= 0):
                        occupied = True
                        break
                self.grid_map.occupancy[i, j] = occupied
        
        rospy.loginfo("Grid Updated.")
        # if self.current_goal is not None:
        #      self.plan()

    def plan(self):
        if self.current_pos is None or self.current_goal is None: return
        
        solver = GraphSearch(self.grid_map)
        path_points = solver.a_star(self.current_pos, self.current_goal)
        
        if not path_points:
            rospy.logwarn("No path found!")
            return

        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = self.frame_id
        
        for pt in path_points:
            pose = PoseStamped()
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        self.pub_path.publish(path_msg)

if __name__ == '__main__':
    try:
        GlobalPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass