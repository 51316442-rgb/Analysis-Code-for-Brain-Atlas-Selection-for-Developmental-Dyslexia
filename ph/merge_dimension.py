import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Merge per-dimension persistence intervals")
parser.add_argument("--input", default="jiangewenjian")
parser.add_argument("--output", default="hebinghoushuju")
args = parser.parse_args()
input_folder = args.input
output_folder = args.output
os.makedirs(output_folder, exist_ok=True)


base_files = [f for f in os.listdir(input_folder) if f.endswith("_W_intervals_0.csv")]

for base_file in base_files:
    # 获取统一前缀,dd_0_pointcloud
    prefix = base_file.replace("_W_intervals_0.csv", "")

    merged_data = []

    for dim in range(3):  # H0, H1, H2
        file_name = f"{prefix}_W_intervals_{dim}.csv"
        file_path = os.path.join(input_folder, file_name)

        if not os.path.exists(file_path):
            print(f"❌ 缺失文件: {file_path}")
            continue
        if os.path.getsize(file_path) == 0:
            print(f"⚠️ 空文件: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path, header=None)
            df['dimension'] = dim
            merged_data.append(df)
        except pd.errors.EmptyDataError:
            print(f"⚠️ 文件无数据: {file_path}")
            continue

    if merged_data:
        merged_df = pd.concat(merged_data, ignore_index=True)
        output_file = os.path.join(output_folder, f"{prefix}_merged.csv")
        merged_df.to_csv(output_file, index=False, header=False)
        print(f"✅ 合并完成: {output_file}")
    else:
        print(f"⚠️ 无有效数据: {prefix}")
