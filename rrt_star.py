import copy
import math
import random
import time
import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
import h5py
import os


class DataRecorder:
    """专业数据记录器，优化存储效率"""

    def __init__(self):
        self.samples = []  # 原始采样点
        self.valid_flags = []  # 有效性标记 (1有效/0无效)
        self.tree_edges = []  # 树连接关系 (parent_idx, child_idx)
        self.paths = []  # 所有优化路径
        self.obstacles = []  # 障碍物信息
        self.map_metadata = {}  # 地图元数据

    def record_environment(self, obstacles, bounds, start, goal):
        """记录完整环境信息"""
        self.obstacles = np.array(obstacles, dtype=np.float32)
        self.map_metadata = {
            "bounds": np.array(bounds, dtype=np.float32),
            "start": np.array(start, dtype=np.float32),
            "goal": np.array(goal, dtype=np.float32),
        }

    def record_runtime_data(self, sample, is_valid, edge=None):
        """高效记录运行时数据"""
        self.samples.append(sample)
        self.valid_flags.append(1 if is_valid else 0)
        if edge:
            self.tree_edges.append(edge)

    def record_path(self, path):
        """记录路径数据"""
        self.paths.append(np.array(path, dtype=np.float32))

    def save_hdf5(self, filename):
        """HDF5高效存储"""
        with h5py.File(filename, "w") as f:
            # 存储原始数据
            f.create_dataset("samples", data=np.array(self.samples), compression="gzip")
            f.create_dataset(
                "valid_flags", data=np.array(self.valid_flags), compression="gzip"
            )
            f.create_dataset(
                "tree_edges", data=np.array(self.tree_edges), compression="gzip"
            )

            # 存储路径数据
            path_grp = f.create_group("paths")
            for i, path in enumerate(self.paths):
                path_grp.create_dataset(f"path_{i}", data=path, compression="gzip")

            # 存储环境信息
            f.create_dataset("obstacles", data=self.obstacles, compression="gzip")
            for k, v in self.map_metadata.items():
                f.attrs[k] = v

            # 生成并存储特征数据
            self._store_derived_features(f)

    def _store_derived_features(self, h5_file):
        """预计算常用特征"""
        # 生成热图
        heatmap = self._generate_heatmap(grid_size=0.5)
        h5_file.create_dataset("heatmap", data=heatmap, compression="gzip")

        # 计算采样点统计特征
        samples = np.array(self.samples)
        valids = np.array(self.valid_flags)
        h5_file.attrs["valid_rate"] = np.mean(valids)
        h5_file.attrs["sample_density"] = len(samples) / (
            (self.map_metadata["bounds"][1] - self.map_metadata["bounds"][0])
            * (self.map_metadata["bounds"][3] - self.map_metadata["bounds"][2])
        )

    def _generate_heatmap(self, grid_size=0.5):
        """生成高斯平滑热图"""
        bounds = self.map_metadata["bounds"]
        x_min, x_max, y_min, y_max = bounds
        x_res = int((x_max - x_min) / grid_size)
        y_res = int((y_max - y_min) / grid_size)

        heatmap = np.zeros((x_res, y_res))
        for path in self.paths:
            indices = ((path - [x_min, y_min]) / grid_size).astype(int)
            valid_indices = (
                (indices[:, 0] >= 0)
                & (indices[:, 0] < x_res)
                & (indices[:, 1] >= 0)
                & (indices[:, 1] < y_res)
            )
            np.add.at(
                heatmap, (indices[valid_indices, 0], indices[valid_indices, 1]), 1
            )

        return gaussian_filter(heatmap, sigma=2)


class Node:
    __slots__ = ["x", "y", "cost", "parent"]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None


class RRTDataGenerator:
    def __init__(
        self,
        obstacle_list,
        area_bounds,
        expand_dis=1.0,
        goal_sample_rate=10,
        max_iter=300,
        data_dir="hgnn_dataset",
    ):
        self.obstacle_list = obstacle_list
        self.area_bounds = area_bounds
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # 初始化记录器和状态变量
        self.recorder = DataRecorder()
        self.node_list = []
        self.node_positions = []
        self.kd_tree = None
        self.goal = None

    def _reset_rrt_state(self, start, goal):
        """重置RRT状态的核心方法"""
        self.node_list = [Node(start[0], start[1])]
        self.node_positions = [[start[0], start[1]]]
        self.kd_tree = KDTree(self.node_positions)
        self.goal = Node(goal[0], goal[1])

    def _is_point_valid(self, x, y):
        """检查坐标点是否在障碍物外"""
        for ox, oy, size in self.obstacle_list:
            if math.hypot(x - ox, y - oy) < size:
                return False
        return True

    def _get_final_path(self):
        """生成最终路径"""
        path = []
        current = self.node_list[-1]
        while current is not None:
            path.append([current.x, current.y])
            current = current.parent
        return path[::-1]

    def _is_near_goal(self, node):
        """目标接近检测"""
        return math.hypot(node.x - self.goal.x, node.y - self.goal.y) <= self.expand_dis

    def _get_nearest_node(self, rnd_point):
        """获取最近节点"""
        _, idx = self.kd_tree.query([rnd_point])
        return self.node_list[idx[0]]

    def _biased_sample(self):
        """智能采样方法，带目标偏向"""
        if random.randint(0, 100) < self.goal_sample_rate:
            return [self.goal.x, self.goal.y]  # 返回目标点坐标
        else:
            # 在自由空间内随机采样
            x = random.uniform(self.area_bounds[0], self.area_bounds[1])
            y = random.uniform(self.area_bounds[2], self.area_bounds[3])
            return [x, y]

    def _steer(self, nearest_node, rnd_point):
        """生成新节点并检测碰撞"""
        theta = math.atan2(rnd_point[1] - nearest_node.y, rnd_point[0] - nearest_node.x)
        new_node = Node(
            nearest_node.x + self.expand_dis * math.cos(theta),
            nearest_node.y + self.expand_dis * math.sin(theta),
        )
        new_node.parent = nearest_node
        new_node.cost = nearest_node.cost + self.expand_dis

        # 碰撞检测
        is_valid = self._check_segment_collision(
            nearest_node.x, nearest_node.y, new_node.x, new_node.y
        )
        return new_node, is_valid

    def _check_segment_collision(self, x1, y1, x2, y2):
        """线段碰撞检测"""
        for ox, oy, size in self.obstacle_list:
            dx = ox - x1
            dy = oy - y1
            fx = x2 - x1
            fy = y2 - y1

            # 计算最小距离
            t = (dx * fx + dy * fy) / (fx**2 + fy**2 + 1e-6)
            t = max(0, min(1, t))
            d_sq = (dx - t * fx) ** 2 + (dy - t * fy) ** 2
            if d_sq < size**2:
                return False
        return True

    def _choose_parent(self, new_node, near_inds):
        """RRT* 父节点选择优化"""
        if not near_inds:
            return

        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            dx = new_node.x - near_node.x
            dy = new_node.y - near_node.y
            if self._check_segment_collision(
                near_node.x, near_node.y, new_node.x, new_node.y
            ):
                costs.append(near_node.cost + math.hypot(dx, dy))
            else:
                costs.append(float("inf"))

        min_cost = min(costs)
        if min_cost < new_node.cost:
            best_idx = near_inds[costs.index(min_cost)]
            new_node.parent = self.node_list[best_idx]
            new_node.cost = min_cost

    def _rewire(self, new_node, near_inds):
        """RRT* 重布线优化"""
        for i in near_inds:
            near_node = self.node_list[i]
            dx = near_node.x - new_node.x
            dy = near_node.y - new_node.y
            tentative_cost = new_node.cost + math.hypot(dx, dy)

            if tentative_cost < near_node.cost:
                if self._check_segment_collision(
                    new_node.x, new_node.y, near_node.x, near_node.y
                ):
                    near_node.parent = new_node
                    near_node.cost = tentative_cost

    def generate_case(self, start, goal, case_id):
        """生成单个案例的完整实现"""
        # 初始化环境记录
        if not self._is_point_valid(*start) or not self._is_point_valid(*goal):
            print(f"Case {case_id}: 起点/终点位于障碍物上，直接放弃")
            return False
        self.recorder = DataRecorder()
        self.recorder.record_environment(
            obstacles=self.obstacle_list,
            bounds=self.area_bounds,
            start=start,
            goal=goal,
        )

        # 重置RRT状态
        self._reset_rrt_state(start, goal)

        # 主算法循环
        for _ in range(self.max_iter):
            # 采样并记录原始点
            rnd = self._biased_sample()
            raw_point = (rnd[0], rnd[1])

            # 路径扩展
            nearest_node = self._get_nearest_node(rnd)
            new_node, is_valid = self._steer(nearest_node, rnd)

            if is_valid:
                # RRT*优化步骤
                self._rrt_star_optimize(new_node)

                # 添加新节点到列表
                self.node_list.append(new_node)
                self.node_positions.append([new_node.x, new_node.y])
                self.kd_tree = KDTree(self.node_positions)

                # 检查是否接近目标
                if self._is_near_goal(new_node):
                    if self._check_segment_collision(
                        new_node.x, new_node.y, self.goal.x, self.goal.y
                    ):
                        final_path = self._get_final_path()
                        final_path.append([self.goal.x, self.goal.y])
                        self.recorder.record_path(final_path)

                # 获取父节点索引
                parent_node = new_node.parent
                parent_index = -1
                if parent_node is not None:
                    try:
                        parent_index = self.node_list.index(parent_node)
                    except ValueError:
                        pass  # 处理异常情况，但理论上不应发生

                # 记录有效样本和边
                child_index = len(self.node_list) - 1
                if parent_index != -1:
                    self.recorder.record_runtime_data(
                        sample=raw_point,
                        is_valid=True,
                        edge=(parent_index, child_index),
                    )
                else:
                    self.recorder.record_runtime_data(
                        sample=raw_point,
                        is_valid=True,
                    )
            else:
                self.recorder.record_runtime_data(sample=raw_point, is_valid=False)

        has_path = len(self.recorder.paths) > 0  # 存在可行路径
        valid_rate = np.mean(self.recorder.valid_flags)
        min_valid_nodes = 30  # 最少需要30个有效采样点

        success = (
            has_path
            and (valid_rate > 0.15)
            and (sum(self.recorder.valid_flags) >= min_valid_nodes)
        )

        if success:
            print(f"Case {case_id}: 有效样本比例={valid_rate*100:.1f}%，保存数据")
            self._save_case_data(case_id)
            return True
        else:
            print(
                f"Case {case_id}: 无效案例（路径存在={has_path}, 有效率={valid_rate:.2f}），放弃并重新生成"
            )
            return False

    def _save_case_data(self, case_id):
        """保存案例数据"""
        filename = os.path.join(self.data_dir, f"case_{case_id:04d}.h5")
        self.recorder.save_hdf5(filename)
        print(f"Saved case {case_id} data: {filename}")

    def _rrt_star_optimize(self, new_node):
        """RRT*优化逻辑"""
        gamma = 5.0  # 理论最优参数
        radius = gamma * math.sqrt(math.log(len(self.node_list)) / len(self.node_list))
        near_inds = self.kd_tree.query_ball_point([new_node.x, new_node.y], radius)

        # 优化父节点选择
        self._choose_parent(new_node, near_inds)
        # 重布线优化
        self._rewire(new_node, near_inds)


def batch_generate(data_dir, num_cases=1000):
    """批量数据生成管道（智能重试版）"""
    # 统一环境参数
    base_obstacles = [
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
    area_bounds = [-2, 18, -2, 15]

    # 创建生成器实例
    generator = RRTDataGenerator(
        obstacle_list=base_obstacles,
        area_bounds=area_bounds,
        data_dir=data_dir,
        expand_dis=1.0,  # 增大扩展距离
        goal_sample_rate=10,  # 提高目标采样率
        max_iter=300,  # 增加最大迭代次数
    )

    success_count = 0
    attempt_count = 0
    max_attempts = num_cases * 4  # 最大允许尝试次数

    def gen_safe_point(is_start):
        """生成安全坐标点（带智能区域划分）"""
        for _ in range(100):  # 最大尝试次数
            if is_start:
                x = random.uniform(
                    area_bounds[0], (area_bounds[0] + area_bounds[1]) * 0.4
                )
                y = random.uniform(
                    area_bounds[2], (area_bounds[2] + area_bounds[3]) * 0.4
                )
            else:
                x = random.uniform(
                    (area_bounds[0] + area_bounds[1]) * 0.6, area_bounds[1]
                )
                y = random.uniform(
                    (area_bounds[2] + area_bounds[3]) * 0.6, area_bounds[3]
                )

            # 安全距离检查（障碍物半径的1.2倍）
            safe = True
            for ox, oy, size in base_obstacles:
                if math.hypot(x - ox, y - oy) < size:
                    safe = False
                    break
            if safe:
                return [x, y]
        return None

    while success_count < num_cases and attempt_count < max_attempts:
        # 生成安全起点终点
        start = gen_safe_point(True)
        goal = gen_safe_point(False)

        # 坐标有效性检查
        if start is None or goal is None:
            print("坐标生成失败，跳过本次尝试")
            attempt_count += 1
            continue

        attempt_count += 1

        # 生成案例（使用success_count作为案例ID）
        if generator.generate_case(start, goal, success_count):
            success_count += 1
            print(f"成功案例进度: {success_count}/{num_cases}")
        else:
            # 失败后直接创建新实例
            generator = RRTDataGenerator(  # 重新初始化生成器
                obstacle_list=base_obstacles,
                area_bounds=area_bounds,
                data_dir=data_dir,
                expand_dis=1.0,
                goal_sample_rate=10,
                max_iter=300,
            )

        # 进度报告
        if attempt_count % 50 == 0:
            current_rate = (
                success_count / attempt_count * 100 if attempt_count > 0 else 0
            )
            print(
                f"尝试次数: {attempt_count} | 成功案例: {success_count} | 成功率: {current_rate:.1f}%"
            )

    # 最终报告
    if success_count < num_cases:
        print(f"⚠️ 生成未完成: {success_count}/{num_cases} 案例")
    else:
        final_rate = success_count / attempt_count * 100 if attempt_count > 0 else 0
        print(f"✅ 生成完成！总尝试 {attempt_count} 次，成功率 {final_rate:.1f}%")


if __name__ == "__main__":
    # 生成1000个训练案例到指定目录
    batch_generate("hgnn_training_data", num_cases=1000)
    print("All data generation completed!")
