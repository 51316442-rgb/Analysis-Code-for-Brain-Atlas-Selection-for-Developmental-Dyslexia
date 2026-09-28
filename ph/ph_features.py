import numpy as np
import pandas as pd
import os
import argparse
from gtda.diagrams import PersistenceEntropy, Amplitude
from gtda.homology import VietorisRipsPersistence
from sklearn.pipeline import make_pipeline, make_union


parser = argparse.ArgumentParser(description="PH feature extraction")
parser.add_argument("--network", default="network(20-back_Power).npy")
parser.add_argument("--output", default="fMRI_features(20-back_Power).csv")
args = parser.parse_args()
network_file = args.network
output_file = args.output
os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

# 加载皮尔逊相关性矩阵
network_data = np.load(network_file)  # shape: (N, n_roi, n_roi)

# 特征提取的度量列表
metric_list = [
    {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 1, "n_bins": 100}},
    {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 2, "n_bins": 100}},
    {"metric": "landscape", "metric_params": {"p": 2, "n_layers": 1, "n_bins": 100}},
    {"metric": "landscape", "metric_params": {"p": 2, "n_layers": 2, "n_bins": 100}},
    {"metric": "betti", "metric_params": {"p": 1, "n_bins": 100}},
    {"metric": "betti", "metric_params": {"p": 2, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 1, "sigma": 1.6, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 1, "sigma": 3.2, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 2, "sigma": 1.6, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 2, "sigma": 3.2, "n_bins": 100}},
]

# 持久性图的计算步骤
diagram_steps = [
    [
        VietorisRipsPersistence(metric='precomputed', n_jobs=-1),  # 使用皮尔逊相关性矩阵计算持久性图
    ]
]

# 组合特征提取器
feature_union = make_union(
    *[PersistenceEntropy(nan_fill_value=4)]
     + [Amplitude(**metric, n_jobs=-1) for metric in metric_list]
)

combined_df = pd.DataFrame()



subject_ids = [f"subject_{i + 1}" for i in range(network_data.shape[0])]

# 遍历每个被试
for idx, network in enumerate(network_data):
    correlation_matrix = network  # shape: (90, 90)

    distance_matrix = 1.0 - np.abs(np.clip(correlation_matrix, -1, 1))
    persistence_diagram = VietorisRipsPersistence(metric='precomputed', n_jobs=-1).fit_transform([distance_matrix])


    features = feature_union.fit_transform(persistence_diagram)


    x_future = pd.DataFrame(features)
    x_future = np.transpose(x_future)

    combined_data = pd.concat([pd.Series([subject_ids[idx]]), x_future], axis=0)
    combined_data = np.transpose(combined_data)
    combined_df = pd.concat([combined_df, combined_data], ignore_index=True)

    print(f"Processed subject {subject_ids[idx]}")

combined_df.columns = ["subject"] + [f"f{i}" for i in range(1, features.shape[1] + 1)]
combined_df.to_csv(output_file, index=False)

print("Feature extraction completed and saved to CSV.")
