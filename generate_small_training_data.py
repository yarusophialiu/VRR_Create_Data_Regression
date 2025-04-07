import pandas as pd
import os
import shutil
from utils import *

# TODO
train_data_folder = f"{VRRML_DATA}/ml_regression/train"
N = 30
mode = 'test' # 'train'

df_patch = pd.read_csv(f"{train_data_folder}/patch_info.csv")
df_patch_small = df_patch.sample(n=N, random_state=42)

# Copy selected patches to new folder
train_data_debug_folder = f"{VRRML_DATA}/ml_regression/debug/{mode}"
dst_dir = f"{train_data_debug_folder}/patches"
os.makedirs(dst_dir, exist_ok=True)

for fname in df_patch_small['patch_path']:
    src = os.path.join(f"{train_data_folder}/patches", fname)
    dst = os.path.join(dst_dir, fname)
    shutil.copyfile(src, dst)

df_patch_small.to_csv(f"{train_data_debug_folder}/patch_info.csv", index=False)

# Get corresponding video IDs and bitrates
df_video = pd.read_csv(f"{train_data_folder}/video_info.csv")
df_video_small = df_video[df_video["video_id"].isin(df_patch_small["video_id"])]
df_video_small.to_csv(f"{train_data_debug_folder}/video_info.csv", index=False)

print(f"Debug dataset saved to ml_regression/debug with {len(df_patch_small)} patches.")
