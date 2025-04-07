import pandas as pd
import os
import shutil
from utils import *
from sklearn.model_selection import train_test_split

# Config
train_data_folder = f"{VRRML_DATA}/ml_regression/all_train_data"
output_base = f"{VRRML_DATA}/ml_regression"
test_size = 0.1  # 30% for validation

# Load and sample from patch_info.csv
df_patch = pd.read_csv(f"{train_data_folder}/patch_info.csv")

df_train, df_val = train_test_split(df_patch, test_size=test_size, random_state=42)

def copy_split(df_patch_split, split_name):
    """Copy patch images and generate split CSVs for train/val."""
    split_folder = os.path.join(output_base, split_name)
    patch_folder = os.path.join(split_folder, "patches")
    os.makedirs(patch_folder, exist_ok=True)

    # Copy patches
    for fname in df_patch_split["patch_path"]:
        src = os.path.join(train_data_folder, "patches", fname)
        dst = os.path.join(patch_folder, fname)
        shutil.copyfile(src, dst)
    
    # selecting only the rows from df_video where the video_id exists in the subset of patch_info.csv
    df_patch_split.to_csv(os.path.join(split_folder, "patch_info.csv"), index=False)

    # Save matching video_info.csv rows
    df_video = pd.read_csv(os.path.join(train_data_folder, "video_info.csv"))
    df_video_split = df_video[df_video["video_id"].isin(df_patch_split["video_id"])]
    df_video_split.to_csv(os.path.join(split_folder, "video_info.csv"), index=False)

    print(f"Saved {split_name} data: {len(df_patch_split)} patches")


copy_split(df_train, "train")
copy_split(df_val, "val")