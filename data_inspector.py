"""
HDF5 Dataset Inspector
Usage: python inspect_hdf5.py <filepath> [--show-samples] [--plot]
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse


def inspect_hdf5(filepath, show_samples=False, plot_data=False):
    """查看HDF5文件内容的核心函数"""
    try:
        with h5py.File(filepath, "r") as f:
            # 显示基础信息
            print(f"\n{'='*40}\nInspecting: {filepath}\n{'='*40}")

            # 1. 显示元数据
            print("\n[Metadata]")
            print(f"Bounds: {f.attrs.get('bounds', 'N/A')}")
            print(f"Start: {f.attrs.get('start', 'N/A')}")
            print(f"Goal: {f.attrs.get('goal', 'N/A')}")
            print(f"Valid Rate: {f.attrs.get('valid_rate', 'N/A'):.2%}")
            print(
                f"Sample Density: {f.attrs.get('sample_density', 'N/A'):.2f} pts/unit²"
            )

            # 2. 显示数据集概览
            print("\n[Dataset Structure]")

            def print_attrs(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  - {name}: {obj.shape} {obj.dtype}")

            f.visititems(print_attrs)

            # 3. 显示采样数据样本
            if show_samples:
                print("\n[Sample Data]")
                samples = f["samples"][:]
                print(f"First 5 samples:\n{samples[:5]}")
                print(f"Validity flags (first 10): {f['valid_flags'][:10].flatten()}")

            # 4. 路径分析
            if "paths" in f:
                print("\n[Path Analysis]")
                for path_name in f["paths"]:
                    path = f[f"paths/{path_name}"][:]
                    print(
                        f"{path_name}: {len(path)} waypoints, "
                        f"Length: {np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)):.2f}"
                    )

            # 5. 可视化数据
            if plot_data:
                plt.figure(figsize=(12, 8))

                # 绘制障碍物
                obstacles = f["obstacles"][:]
                for x, y, r in obstacles:
                    circle = plt.Circle((x, y), r, color="gray", alpha=0.5)
                    plt.gca().add_patch(circle)

                # 绘制采样点
                samples = f["samples"][:]
                valid = f["valid_flags"][:].flatten().astype(bool)
                plt.scatter(
                    samples[valid, 0],
                    samples[valid, 1],
                    c="green",
                    s=10,
                    label="Valid Samples",
                    alpha=0.6,
                )
                plt.scatter(
                    samples[~valid, 0],
                    samples[~valid, 1],
                    c="red",
                    s=10,
                    label="Invalid Samples",
                    alpha=0.6,
                )

                # 绘制路径
                if "paths" in f:
                    for path_name in f["paths"]:
                        path = f[f"paths/{path_name}"][:]
                        plt.plot(
                            path[:, 0],
                            path[:, 1],
                            linewidth=2,
                            linestyle="--",
                            label=f"Path {path_name}",
                        )

                # 绘制热力图
                heatmap = f["heatmap"][:]
                bounds = f.attrs["bounds"]
                x_grid = np.linspace(bounds[0], bounds[1], heatmap.shape[0])
                y_grid = np.linspace(bounds[2], bounds[3], heatmap.shape[1])
                plt.contourf(
                    x_grid, y_grid, heatmap.T, levels=20, cmap="YlOrRd", alpha=0.3
                )

                plt.title(f"Case Visualization: {filepath}")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.axis("equal")
                plt.legend()
                plt.grid(True)
                plt.show()

    except Exception as e:
        print(f"Error inspecting file: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDF5 Dataset Inspector")
    parser.add_argument("filepath", help="Path to HDF5 file")
    parser.add_argument(
        "--show-samples", action="store_true", help="Display sample data points"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate visualization plot"
    )
    args = parser.parse_args()

    inspect_hdf5(args.filepath, args.show_samples, args.plot)
