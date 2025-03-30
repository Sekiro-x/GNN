import torch
import random
import numpy as np
from torch_geometric.data import HeteroData
from train_hgnn import HeteroRRT
from rrt_base import RRT, Node  # 从基础文件导入原始RRT类

show_animation = True  # 是否显示动画


class HGNNSampler:
    """HGNN模型推理工具类"""

    def __init__(self, obstacle_list, bounds):
        # 加载训练好的模型
        self.model = HeteroRRT(hidden_channels=128, num_heads=4)  # 参数必须与训练时一致
        self.model.load_state_dict(
            torch.load("hgnn_final_model.pth", map_location="cpu")
        )
        self.model.eval()

        # 场景边界参数
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        self.max_range = max(self.x_range, self.y_range)

        # 障碍物信息
        self.obstacle_list = obstacle_list

    def _normalize_coord(self, coord):
        """坐标归一化（与训练时保持一致）"""
        return [
            (coord[0] - self.x_min) / self.x_range,
            (coord[1] - self.y_min) / self.y_range,
        ]

    def build_hetero_data(self, tree_nodes, start, goal):
        """构建异质图输入数据"""
        data = HeteroData()

        # 处理样本节点
        sample_coords = np.array([[n.x, n.y] for n in tree_nodes])
        if len(sample_coords) == 0:
            return None

        # 归一化处理（与训练代码完全一致）
        sample_norm = np.array(
            [self._normalize_coord((x, y)) for x, y in sample_coords]
        )
        valid_flags = np.ones(len(sample_coords))  # 假设所有树节点都有效

        # 障碍物处理
        obstacle_coords = np.array([(x, y) for x, y, _ in self.obstacle_list])
        obstacle_radii = np.array(
            [r / self.max_range for _, _, r in self.obstacle_list]
        )
        obstacle_norm = np.array(
            [self._normalize_coord((x, y)) for x, y in obstacle_coords]
        )

        # 构建节点特征
        data["SampleNode"].x = torch.tensor(
            np.hstack([sample_norm, valid_flags.reshape(-1, 1)]), dtype=torch.float32
        )
        data["ObstacleNode"].x = torch.tensor(
            np.hstack([obstacle_norm, obstacle_radii.reshape(-1, 1)]),
            dtype=torch.float32,
        )
        data["StartNode"].x = torch.tensor(
            [self._normalize_coord(start)], dtype=torch.float32
        )
        data["GoalNode"].x = torch.tensor(
            [self._normalize_coord(goal)], dtype=torch.float32
        )

        # 构建树结构边
        if len(tree_nodes) > 0:
            parent_indices = [n.parent for n in tree_nodes if n.parent is not None]
            child_indices = [
                i for i, n in enumerate(tree_nodes) if n.parent is not None
            ]
            if parent_indices:
                tree_edges = torch.tensor(
                    [parent_indices, child_indices], dtype=torch.long
                )
                data["SampleNode", "tree_edge", "SampleNode"].edge_index = tree_edges

        # 构建障碍物邻近边
        sample_tensor = torch.tensor(sample_norm)
        obstacle_tensor = torch.tensor(obstacle_norm)
        dist_matrix = torch.cdist(sample_tensor, obstacle_tensor, p=2)
        thresholds = 2.0 * torch.tensor(obstacle_radii)
        mask = dist_matrix <= thresholds
        src, dst = torch.where(mask)
        if len(src) > 0:
            data["SampleNode", "near_obstacle", "ObstacleNode"].edge_index = (
                torch.stack([src, dst], 0)
            )

        return data


class HGNNRRT(RRT):
    """基于HGNN的改进RRT*算法"""

    def __init__(
        self, obstacleList, randArea, expandDis=1.0, goalSampleRate=15, maxIter=300
    ):
        super().__init__(obstacleList, randArea, expandDis, goalSampleRate, maxIter)

        # 初始化HGNN采样器
        self.sampler = HGNNSampler(
            obstacle_list=obstacleList,
            bounds=[randArea[0], randArea[1], randArea[0], randArea[1]],
        )
        self.k_candidates = 50  # 每轮采样候选点数

    def sample(self):
        """基于HGNN的智能采样"""
        if random.randint(0, 100) < self.goal_sample_rate:
            return [self.goal.x, self.goal.y]

        # 生成候选点
        candidates = np.array(
            [
                [
                    random.uniform(self.min_rand, self.max_rand),
                    random.uniform(self.min_rand, self.max_rand),
                ]
                for _ in range(self.k_candidates)
            ]
        )

        # 获取模型预测
        data = self.sampler.build_hetero_data(
            self.node_list, [self.start.x, self.start.y], [self.goal.x, self.goal.y]
        )
        if data is None:
            return super().sample()

        with torch.no_grad():
            logits = self.sampler.model(data)
            probs = torch.sigmoid(logits).numpy()

        # 选择概率最高的候选点
        if len(probs) == len(candidates):
            best_idx = np.argmax(probs)
            return candidates[best_idx]
        return super().sample()  # 异常时退回随机采样


def main():
    print("Running HGNN-enhanced RRT*")

    # 障碍物配置（与原始实验相同）
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

    # 初始化算法
    hgnn_rrt = HGNNRRT(
        obstacleList=obstacleList, randArea=[-2, 18], expandDis=1.0, maxIter=300
    )

    # 路径规划
    path = hgnn_rrt.rrt_star_planning(start=[0, 0], goal=[15, 12], animation=True)

    print("Planning completed!")
    if path and show_animation:
        plt.show()


if __name__ == "__main__":
    main()
