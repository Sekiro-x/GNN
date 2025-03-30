import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData, Batch
from torch_geometric.loader import DataLoader
import h5py
import os
import numpy as np


# -------------------------- 自定义数据集 -------------------------- #
class RRTDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.filenames = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".h5") and os.path.isfile(os.path.join(data_dir, f))
        ]
        self.cache = {}  # 添加缓存加速读取

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        if index not in self.cache:
            self.cache[index] = self.h5_to_heterograph(self.filenames[index])
        return self.cache[index]

    def h5_to_heterograph(self, filename):
        with h5py.File(filename, "r") as f:
            samples = f["samples"][()].astype(np.float32)
            valid_flags = f["valid_flags"][()]
            tree_edges = f["tree_edges"][()].astype(np.int64)
            obstacles = f["obstacles"][()].astype(np.float32)
            start = f.attrs["start"].astype(np.float32)
            goal = f.attrs["goal"].astype(np.float32)
            bounds = f.attrs["bounds"].astype(np.float32)

        data = HeteroData()

        # 归一化处理（包含障碍物半径）
        x_min, x_max, y_min, y_max = bounds
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range)

        # 样本点归一化
        samples[:, 0] = (samples[:, 0] - x_min) / x_range
        samples[:, 1] = (samples[:, 1] - y_min) / y_range

        # 障碍物归一化（坐标+半径）
        obstacles[:, 0] = (obstacles[:, 0] - x_min) / x_range
        obstacles[:, 1] = (obstacles[:, 1] - y_min) / y_range
        obstacles[:, 2] /= max_range  # 半径按最大范围归一化

        # 起点终点归一化
        start = np.array(
            [(start[0] - x_min) / x_range, (start[1] - y_min) / y_range],
            dtype=np.float32,
        )

        goal = np.array(
            [(goal[0] - x_min) / x_range, (goal[1] - y_min) / y_range], dtype=np.float32
        )

        # 构建节点特征
        sample_features = np.zeros((len(samples), 3), dtype=np.float32)
        sample_features[:, :2] = samples[:, :2]
        sample_features[:, 2] = valid_flags
        data["SampleNode"].x = torch.tensor(sample_features)
        data["SampleNode"].y = torch.tensor(valid_flags, dtype=torch.float32).view(
            -1, 1
        )

        data["ObstacleNode"].x = torch.tensor(obstacles)
        data["StartNode"].x = torch.tensor(start[None, :])
        data["GoalNode"].x = torch.tensor(goal[None, :])

        # 构建边关系（添加对称障碍物边）
        samples_t = data["SampleNode"].x[:, :2]
        obstacles_t = data["ObstacleNode"].x[:, :2]
        obs_sizes = data["ObstacleNode"].x[:, 2]

        # 使用PyG内置函数计算距离
        distance_matrix = torch.cdist(samples_t, obstacles_t, p=2)
        thresholds = 2.0 * obs_sizes.view(1, -1)
        mask = distance_matrix <= thresholds

        # 构建双向障碍物边
        src, dst = mask.nonzero(as_tuple=True)
        obs_edge_index = torch.stack([src, dst], dim=0)
        data["SampleNode", "near_obstacle", "ObstacleNode"].edge_index = obs_edge_index
        data["ObstacleNode", "near_sample", "SampleNode"].edge_index = (
            obs_edge_index.flip([0])
        )

        # 树结构边
        if len(tree_edges) > 0:
            tree_edge_index = (
                torch.tensor(tree_edges, dtype=torch.long).t().contiguous()
            )
            data["SampleNode", "tree_edge", "SampleNode"].edge_index = tree_edge_index

        # 全局连接边（起点到所有样本）
        n_samples = len(samples)
        global_edge_index = torch.tensor(
            [[0] * n_samples, range(n_samples)], dtype=torch.long  # start -> samples
        )
        data["StartNode", "global_edge", "SampleNode"].edge_index = global_edge_index

        return data


# -------------------------- HGNN模型定义 -------------------------- #
class HeteroRRT(nn.Module):
    def __init__(self, hidden_channels=128, num_heads=4):
        super().__init__()

        # 增强的特征嵌入
        self.embed = nn.ModuleDict(
            {
                "SampleNode": nn.Sequential(
                    nn.Linear(3, hidden_channels), nn.LayerNorm(hidden_channels)
                ),
                "ObstacleNode": nn.Sequential(
                    nn.Linear(3, hidden_channels), nn.LayerNorm(hidden_channels)
                ),
                "StartNode": nn.Sequential(
                    nn.Linear(2, hidden_channels), nn.LayerNorm(hidden_channels)
                ),
                "GoalNode": nn.Sequential(
                    nn.Linear(2, hidden_channels), nn.LayerNorm(hidden_channels)
                ),
            }
        )

        # 异构图元数据
        node_types = ["SampleNode", "ObstacleNode", "StartNode", "GoalNode"]
        edge_types = [
            # 原始边
            ("SampleNode", "tree_edge", "SampleNode"),
            ("SampleNode", "near_obstacle", "ObstacleNode"),
            ("ObstacleNode", "near_sample", "SampleNode"),
            ("StartNode", "global_edge", "SampleNode"),
            # 添加逆向边
            ("SampleNode", "rev_tree_edge", "SampleNode"),
            ("ObstacleNode", "rev_near_obstacle", "SampleNode"),
            ("SampleNode", "rev_near_sample", "ObstacleNode"),
            ("SampleNode", "rev_global_edge", "StartNode"),
        ]
        metadata = (node_types, edge_types)

        # 使用3层HGT卷积
        self.conv1 = HGTConv(
            hidden_channels, hidden_channels, metadata, heads=num_heads
        )
        self.conv2 = HGTConv(
            hidden_channels, hidden_channels, metadata, heads=num_heads
        )
        self.conv3 = HGTConv(
            hidden_channels, hidden_channels, metadata, heads=num_heads
        )

        # 残差连接层
        self.res_linear = nn.ModuleDict(
            {
                "SampleNode": nn.Linear(hidden_channels, hidden_channels),
                "ObstacleNode": nn.Linear(hidden_channels, hidden_channels),
                "StartNode": nn.Identity(),  # 新增
                "GoalNode": nn.Identity(),  # 新增
            }
        )

        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, data):
        x_dict = {nt: self.embed[nt](data.x_dict[nt]) for nt in data.x_dict}

        # 保留原始节点类型集合
        original_node_types = set(x_dict.keys())

        # 第一层卷积
        x_dict_conv1 = self.conv1(x_dict, data.edge_index_dict)
        # 保留所有节点类型
        x_dict = {k: x_dict_conv1.get(k, x_dict[k]) for k in original_node_types}
        x_dict = {k: F.leaky_relu(v, 0.2) for k, v in x_dict.items()}

        # 第二层卷积
        x_dict_conv2 = self.conv2(x_dict, data.edge_index_dict)
        # 残差连接保留所有节点类型
        x_dict = {
            k: F.leaky_relu(
                x_dict_conv2.get(k, torch.zeros_like(x_dict[k]))
                + self.res_linear[k](x_dict[k]),
                0.2,
            )
            for k in original_node_types
        }

        # 第三层卷积
        x_dict_conv3 = self.conv3(x_dict, data.edge_index_dict)
        # 保留所有节点类型
        x_dict = {k: x_dict_conv3.get(k, x_dict[k]) for k in original_node_types}
        x_dict = {k: F.leaky_relu(v, 0.2) for k, v in x_dict.items()}

        return self.output(x_dict["SampleNode"]).squeeze()


# -------------------------- 训练流程 -------------------------- #
def train():
    # 配置参数
    data_dir = "hgnn_training_data"
    batch_size = 32
    num_epochs = 100
    hidden_channels = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集划分
    full_dataset = RRTDataset(data_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # 数据加载器（修正collate_fn）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=Batch.from_data_list,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=Batch.from_data_list,
        num_workers=4,
    )

    # 初始化模型
    model = HeteroRRT(hidden_channels=hidden_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    loss_fn = nn.BCEWithLogitsLoss()

    # 训练循环
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            mixed_precision_dtype = torch.float16
            with torch.amp.autocast(
                device_type="cuda", dtype=mixed_precision_dtype
            ):  # 混合精度训练
                pred = model(batch)
                loss = loss_fn(pred, batch["SampleNode"].y.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()

            train_loss += loss.item()

        # 验证阶段
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = loss_fn(pred, batch["SampleNode"].y.view(-1))
                val_loss += loss.item()

        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), f"best_model_epoch{epoch}.pth")
        else:
            early_stop_counter += 1

        # 打印日志
        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if early_stop_counter >= 10:
            print("Early stopping triggered!")
            break

    # 保存最终模型
    torch.save(model.state_dict(), "hgnn_final_model.pth")


if __name__ == "__main__":
    train()
