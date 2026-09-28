import numpy as np
from ripser import ripser  #ripser用于计算持续同调
from persim import plot_diagrams  #persim用于可视化持续图
import os
import argparse


def witness_ripser(filename, max_dimension, num_landmark_points, point_selector, filtr_steps, output_dir):
    """
    :param filename: Path to the text file with point cloud, one point per line.
    :param max_dimension: Maximum dimension for the simplicial complex.
    :param num_landmark_points: Number of landmark points.
    :param point_selector: 'random' or 'maxmin' for selecting landmark points.
    """

    point_cloud = np.loadtxt(filename, delimiter=',', skiprows=1)  #从文件加载点云数据到NumPy数组中，假设每行一个点，逗号分隔坐标

    num_points = point_cloud.shape[0]
    if filtr_steps > num_points:
        filtr_steps = num_points

    if point_selector == 'random':
        np.random.seed(0)
        indices = np.random.choice(num_points, num_landmark_points, replace=False)
    elif point_selector == 'maxmin':
        from sklearn.metrics import pairwise_distances
        dist_matrix = pairwise_distances(point_cloud)
        indices = np.zeros(num_landmark_points, dtype=int)
        indices[0] = np.argmax(np.min(dist_matrix, axis=0))
        for i in range(1, num_landmark_points):
            dist_to_selected = dist_matrix[indices[:i], :]
            min_dist_to_selected = np.min(dist_to_selected, axis=0)
            indices[i] = np.argmax(min_dist_to_selected)
    else:
        raise ValueError("Invalid point_selector. Use 'random' or 'maxmin'.")

    landmark_points = point_cloud[indices]

    result = ripser(landmark_points, maxdim=max_dimension, n_perm=filtr_steps)

    os.makedirs(output_dir, exist_ok=True)
    for dim, intervals in enumerate(result['dgms']):
        os.path.splitext(os.path.basename(filename))[0]
        output_intervals = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(filename))[0]}_W_intervals_{dim}.csv")
        np.savetxt(output_intervals, intervals, delimiter=',', fmt='%f')

    print("Persistence intervals saved.")

def make_dir_witness_ripser(folder_path, output_folder):
    max_dimension = 2
    num_landmark_points = 90
    point_selector = 'maxmin'  # Or 'random'
    filtr_steps = 90
    # 遍历文件夹中的所有CSV文件
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            witness_ripser(file_path, max_dimension, num_landmark_points, point_selector, filtr_steps, output_folder)

parser = argparse.ArgumentParser(description="Witness-complex PH via ripser")
parser.add_argument("--input", default="point_cloud",)
parser.add_argument("--output", default="jiangewenjian",)
args = parser.parse_args()
filename = args.input
output_dir = args.output
make_dir_witness_ripser(filename, output_dir)


# # Example usage
# filename = 'point_cloud.csv'
# max_dimension = 2
# num_landmark_points = 90
# point_selector = 'random'
# filtr_steps = 90
# output_dir = ''
#
# witness_ripser(filename, max_dimension, num_landmark_points, point_selector, filtr_steps, output_dir)
