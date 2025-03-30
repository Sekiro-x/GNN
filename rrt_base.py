import copy
import math
import random
import time

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import numpy as np

show_animation = True


class RRT:

    def __init__(
        self, obstacleList, randArea, expandDis=1.0, goalSampleRate=10, maxIter=200
    ):

        self.start = None
        self.goal = None
        self.min_rand = randArea[0]
        self.max_rand = randArea[1]
        self.expand_dis = expandDis
        self.goal_sample_rate = goalSampleRate
        self.max_iter = maxIter
        self.obstacle_list = obstacleList
        self.node_list = None
        self.valid_samples = 0

    def rrt_star_planning(self, start, goal, animation=True):
        start_time = time.time()
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.node_list = [self.start]
        path = None
        lastPathLength = float("inf")
        self.valid_samples = 0

        for i in range(self.max_iter):
            rnd = self.sample()
            n_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearestNode = self.node_list[n_ind]

            # Steer with adaptive step size (optional)
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)
            newNode = self.get_new_node(theta, n_ind, nearestNode)

            if self.check_segment_collision(
                nearestNode.x, nearestNode.y, newNode.x, newNode.y
            ):
                self.valid_samples += 1
                # Adaptive neighborhood radius
                n_node = len(self.node_list)
                gamma = 8  # Adjust based on scenario
                r = (
                    gamma * math.sqrt(math.log(n_node) / n_node)
                    if n_node > 1
                    else gamma
                )

                nearInds = self.find_near_nodes(newNode, r)
                newNode = self.choose_parent(newNode, nearInds)
                self.node_list.append(newNode)
                self.rewire(newNode, nearInds)

                if animation:
                    self.draw_graph(newNode, path)

                if self.is_near_goal(newNode):
                    if self.check_segment_collision(
                        newNode.x, newNode.y, self.goal.x, self.goal.y
                    ):
                        lastIndex = len(self.node_list) - 1
                        tempPath = self.get_final_course(lastIndex)
                        tempPathLen = self.get_path_len(tempPath)
                        if lastPathLength > tempPathLen:
                            path = tempPath
                            lastPathLength = tempPathLen
                            print(
                                f"Path improved to {tempPathLen:.5f}, Time: {time.time()-start_time:.5f}s"
                            )
        if path:
            print("\n===== Final Statistics =====")
            print(f"Total iterations: {self.max_iter}")
            print(f"Valid samples: {self.valid_samples}")
            print(f"Tree height: {self.calculate_tree_height()}")
            print(f"Optimal path height: {len(path)-1} nodes")
            print(f"Optimal path length: {lastPathLength:.6f}")
        return path

    def sample(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand),
            ]
        else:  # goal point sampling
            rnd = [self.goal.x, self.goal.y]
        return rnd

    def choose_parent(self, newNode, nearInds):
        if len(nearInds) == 0:
            return newNode

        dList = []
        for i in nearInds:
            dx = newNode.x - self.node_list[i].x
            dy = newNode.y - self.node_list[i].y
            d = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            if self.check_collision(self.node_list[i], theta, d):
                dList.append(self.node_list[i].cost + d)
            else:
                dList.append(float("inf"))

        minCost = min(dList)
        minInd = nearInds[dList.index(minCost)]

        if minCost == float("inf"):
            print("min cost is inf")
            return newNode

        newNode.cost = minCost
        newNode.parent = minInd

        return newNode

    def find_near_nodes(self, newNode, radius):
        d_list = [
            (node.x - newNode.x) ** 2 + (node.y - newNode.y) ** 2
            for node in self.node_list
        ]
        near_inds = [i for i, d in enumerate(d_list) if d <= radius**2]
        return near_inds

    @staticmethod
    def get_path_len(path):
        pathLen = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            pathLen += math.sqrt((node1_x - node2_x) ** 2 + (node1_y - node2_y) ** 2)

        return pathLen

    @staticmethod
    def line_cost(node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    @staticmethod
    def get_nearest_list_index(nodes, rnd):
        dList = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in nodes]
        minIndex = dList.index(min(dList))
        return minIndex

    def get_new_node(self, theta, n_ind, nearestNode):
        newNode = copy.deepcopy(nearestNode)

        newNode.x += self.expand_dis * math.cos(theta)
        newNode.y += self.expand_dis * math.sin(theta)

        newNode.cost += self.expand_dis
        newNode.parent = n_ind
        return newNode

    def is_near_goal(self, node):
        d = self.line_cost(node, self.goal)
        if d < self.expand_dis:
            return True
        return False

    def rewire(self, newNode, nearInds):
        for i in nearInds:
            nearNode = self.node_list[i]
            dx = newNode.x - nearNode.x
            dy = newNode.y - nearNode.y
            d = math.hypot(dx, dy)
            if d == 0:
                continue
            if self.check_segment_collision(
                nearNode.x, nearNode.y, newNode.x, newNode.y
            ):
                s_cost = newNode.cost + d
                if nearNode.cost > s_cost:
                    nearNode.parent = len(self.node_list) - 1  # newNode's index
                    nearNode.cost = s_cost

    @staticmethod
    def distance_squared_point_to_segment(v, w, p):
        # Return minimum distance between line segment vw and point p
        if np.array_equal(v, w):
            return (p - v).dot(p - v)  # v == w case
        l2 = (w - v).dot(w - v)  # i.e. |w-v|^2 -  avoid a sqrt
        # Consider the line extending the segment,
        # parameterized as v + t (w - v).
        # We find projection of point p onto the line.
        # It falls where t = [(p-v) . (w-v)] / |w-v|^2
        # We clamp t from [0,1] to handle points outside the segment vw.
        t = max(0, min(1, (p - v).dot(w - v) / l2))
        projection = v + t * (w - v)  # Projection falls on the segment
        return (p - projection).dot(p - projection)

    def check_segment_collision(self, x1, y1, x2, y2):
        for ox, oy, size in self.obstacle_list:
            dd = self.distance_squared_point_to_segment(
                np.array([x1, y1]), np.array([x2, y2]), np.array([ox, oy])
            )
            if dd <= size**2:
                return False  # collision
        return True

    def check_collision(self, nearNode, theta, d):
        tmpNode = copy.deepcopy(nearNode)
        end_x = tmpNode.x + math.cos(theta) * d
        end_y = tmpNode.y + math.sin(theta) * d
        return self.check_segment_collision(tmpNode.x, tmpNode.y, end_x, end_y)

    def get_final_course(self, lastIndex):
        path = []
        current_index = lastIndex
        while current_index is not None:
            node = self.node_list[current_index]
            path.append([node.x, node.y])
            current_index = node.parent
        path = path[::-1]
        if self.check_segment_collision(
            path[-1][0], path[-1][1], self.goal.x, self.goal.y
        ):
            path.append([self.goal.x, self.goal.y])
        return path

    def calculate_tree_height(self):
        depths = {}
        max_depth = 0
        for node in self.node_list:
            depth = 0
            current = node
            while current.parent is not None:
                depth += 1
                current = self.node_list[current.parent]
            max_depth = max(max_depth, depth)
        return max_depth

    def draw_graph(self, rnd=None, path=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")

        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot(
                        [node.x, self.node_list[node.parent].x],
                        [node.y, self.node_list[node.parent].y],
                        "-g",
                    )

        for ox, oy, size in self.obstacle_list:
            # self.plot_circle(ox, oy, size)
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")

        if path is not None:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], "-r")

        plt.axis([-2, 18, -2, 15])
        plt.grid(True)
        plt.pause(0.01)


class Node:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None


def main():
    print("Start planning")

    obstacleList = [
        (2, 2, 1),
        (8, 1, 2),
        (5, 2, 2),
        (3, 1, 1),
        (14, 8, 1),
        (0, 14, 1),
        (7, 6, 2),
        (4, 7, 1),
        (10, 3, 1),
        (-2, 6, 1),
    ]

    # Set params
    rrt = RRT(randArea=[-2, 18], obstacleList=obstacleList, maxIter=300)
    path = rrt.rrt_star_planning(start=[0, 0], goal=[15, 12], animation=show_animation)

    print("Done!!")

    if show_animation and path:
        plt.show()


if __name__ == "__main__":
    main()
