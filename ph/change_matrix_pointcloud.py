import os
import argparse
import numpy as np
import pandas as pd
from sklearn.manifold import Isomap


parser = argparse.ArgumentParser(description="Distance matrix -> 3D point cloud (ISOMAP)")
parser.add_argument("--input", default="PCC")
parser.add_argument("--output", default="point_cloud")
args = parser.parse_args()
folder_path = args.input
output_folder = args.output

def generate_point_cloud(distance_matrix_file, output_folder):
    isomap = Isomap(n_components=3)

    distance_matrix = pd.read_csv(distance_matrix_file).values  #.values:将Pandas DataFrame转换为NumPy数组，以便Isomap可以处理
    point_cloud = isomap.fit_transform(distance_matrix)

    os.makedirs(output_folder, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(distance_matrix_file))[0]
    output_file = os.path.join(output_folder, file_name + "_pointcloud.csv")

    df = pd.DataFrame(point_cloud, columns=["x", "y", "z"])
    df.to_csv(output_file, index=False)

def process_distance_matrix_files(folder_path, output_folder):
    for file in os.listdir(folder_path):

        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            generate_point_cloud(file_path, output_folder)

process_distance_matrix_files(folder_path, output_folder)
print('-----')