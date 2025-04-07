import os
import cv2
import random
import numpy as np
import pandas as pd
from utils import *
from datetime import datetime



def gather_mp4_files(root_dir):
    """
    Recursively collect all .mp4 files paths under root_dir.
    Returns a list of absolute file paths.
    """
    mp4_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".mp4"):
                full_path = os.path.join(dirpath, fname)
                mp4_files.append(full_path)
    return mp4_files

def extract_random_patch(frame, patch_size=128):
    """
    Given a BGR frame from OpenCV (shape [H, W, 3]), 
    extract a random patch of shape (patch_size x patch_size).
    Returns the cropped patch (same 3 channels, BGR).
    """
    h, w, _ = frame.shape

    if h < patch_size or w < patch_size:
        # If frame is too small, just return the center or skip.
        start_y = max(0, h//2 - patch_size//2)
        start_x = max(0, w//2 - patch_size//2)
    else:
        start_y = random.randint(0, h - patch_size)
        start_x = random.randint(0, w - patch_size)

    patch = frame[start_y:start_y+patch_size, start_x:start_x+patch_size, :]
    return patch

def parse_video_id_from_path(video_path):
    """
    Example function to parse a 'video_id' from a path string.
    We take the last 2-3 directory names plus the file name minus extension.
    Customize as needed!
    E.g. reference/bedroom/bedroom_path1_seg1_1/ref166_1080/refOutput.mp4
         -> "bedroom_path1_seg1_1_ref166_1080_refOutput"
    """
    parts = video_path.split(os.sep)
    # For a path like: [reference, bedroom, bedroom_path1_seg1_1, ref166_1080, refOutput.mp4]
    # we can combine the last few segments for a unique ID:
    if len(parts) >= 3:
        # folder1 = parts[-2]       # e.g. "ref166_1080"
        folder2 = parts[-3]       # e.g. "bedroom_path1_seg1_1"
        # filename = os.path.splitext(parts[-1])[0]  # "refOutput"
        video_id = f"{folder2}"
    else:
        # fallback if path shorter than expected
        filename = os.path.splitext(parts[-1])[0]
        video_id = filename
    return video_id



def read_frame_velocity(frame_velocity_path, frame_number):
    with open(frame_velocity_path, "r") as file:
        for line in file:
            frame, velocity = line.split()
            if int(frame) == frame_number:
                return float(velocity)
        return None  # Return None if frame_number is not found


def find_row_dict(scene, sheet_name):
    """sheet_name is like path1_seg1_1"""
    file_path = f'{JOD_CSV}/{scene}.xlsx'
    xls = pd.ExcelFile(file_path)
    all_jod_entries = []

    print(f'sheet_name {sheet_name}')
    df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    resolutions = [i for _ in range(10) for i in [360, 480, 720, 864, 1080]]
    fps_headers = [i for i in range(30, 121, 10) for _ in range(5)]

    # print(f'resolutions {resolutions}')
    # print(f'fps_headers {fps_headers}')

    data = df.iloc[1:, :].reset_index(drop=True)
    # print(f'data {data}')
    column_names  = ['bitrate'] + [f"{res}-fps{fps}" for res, fps in zip(resolutions, fps_headers)]
    # Extract data (skip the first row which is header-like)
    data = df.iloc[1:, :len(column_names)].reset_index(drop=True)
    data.columns = column_names

    # Clean up: convert bitrate column
    data['bitrate'] = pd.to_numeric(data['bitrate'], errors='coerce')
    data = data.dropna(subset=['bitrate']).copy()
    data['bitrate'] = data['bitrate'].astype(int)
    
    if data.empty:
        print(f"[Warning] No valid bitrate rows in {sheet_name} of {scene}")
        return None

    for i in range(4):
        best_row = data.loc[i]
        bitrate = best_row['bitrate']

        print(f'bitrate {bitrate}')
        jod_values = best_row[1:].tolist()
        print(f'jod_values {jod_values}')

        row_dict = {
            'video_id': f"{scene}_{sheet_name}",
            'bitrate': bitrate
        }
        for i, jod in enumerate(jod_values):
            row_dict[f'jod_{i}'] = float(jod)
        # print(f'row_dict {row_dict}')
    return row_dict

def main(scene, reference_dir="reference", 
         patches_output_dir="patches",
         video_info_csv="video_info.csv",
         patch_info_csv="patch_info.csv",
         patch_size=128):
    """
    1) Gather all MP4 videos from reference_dir.
    2) For each video:
       - parse video_id
       - write a row in video_info for the (video_id, random 50 JOD)
       - open the video, extract random patches
       - write a row in patch_info for each patch
    """

    video_paths = gather_mp4_files(reference_dir) # len 450

    video_info_rows = []
    patch_info_rows = []

    # os.makedirs(patches_output_dir, exist_ok=True)
    count = 0
    for vid_path in video_paths:
        video_id = parse_video_id_from_path(vid_path) # like bedroom_path1_seg1_2
        print(f'video_id {video_id}')
        
        # For demonstration, generate random JOD scores and random "bitrate"
        random_jod = np.random.uniform(low=1.0, high=5.0, size=(50,))
        random_bitrate = random.randint(500, 2500)  # just a dummy example

        # Store in video_info
        row_dict = {
            "video_id": video_id,
            "bitrate": random_bitrate
        }

        path_info = video_id.split("_path", 1)[1]  # Split only at the first "_path"
        path_info = "path" + path_info 
        row_dict = find_row_dict(scene, path_info)

        jod_df = pd.DataFrame(row_dict)
        print(jod_df.head())
        jod_df.to_csv(video_info_csv, index=False)

        # video_info_rows.append(row_dict)
        random_jod = np.random.uniform(low=1.0, high=5.0, size=(50,))
        for i in range(50):
            row_dict[f"jod_{i}"] = float(random_jod[i])
        video_info_rows.append(row_dict)

        print(f'row_dict {row_dict}')

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video: {vid_path}")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'frame_count {frame_count}')
        if frame_count == 0:
            print(f"Warning: Zero frames in video: {vid_path}")
            cap.release()
            continue
        frame_velocity_path = f'{VRR_Motion}/reference/magnitude_motion_per_frame/{scene}/{video_id}_velocity_per_frame.txt'

        # We'll extract `patches_per_video` random patches from random frames
        frame_generated = 0
        frame_number = 0 # decoded video frame index that will be passed to find_motion_patch_h265
        while cap.isOpened(): # Read until video is completed
            print(f'frame_number {frame_number}')
            ret, frame = cap.read() # (360, 640, 3)
            if not ret:
                continue
            
            patch_bgr = extract_random_patch(frame, patch_size=patch_size)
            

            frame_velocity = read_frame_velocity(frame_velocity_path, frame_number)
            print(f'random_velocity {frame_velocity}')
            if frame_velocity is None:
                frame_generated += 1
                frame_number += 1
                continue

            patch_filename = f"{video_id}_frame{frame_number}.jpg"
            # os.makedirs(patches_output_dir, exist_ok=True)
            patch_path_full = os.path.join(patches_output_dir, patch_filename)
            cv2.imwrite(patch_path_full, patch_bgr)  # BGR -> writes as JPEG

            # Add row to patch_info
            patch_info_rows.append({
                "patch_path": patch_filename,
                "video_id": video_id,
                "velocity": frame_velocity
            })

            frame_number += 1
            if frame_number == 5:
                break
        cap.release()
        count += 1
        # if count >= 2:
        #     break
    # os.makedirs(patches_output_dir, exist_ok=True)
    # 3. Convert rows to DataFrame, then save as CSV
    video_info_df = pd.DataFrame(video_info_rows)
    video_info_df.to_csv(video_info_csv, index=False)
    print(f"Saved video_info to '{video_info_csv}' with {len(video_info_df)} entries.")

    patch_info_df = pd.DataFrame(patch_info_rows)
    patch_info_df.to_csv(patch_info_csv, index=False)
    print(f"Saved patch_info to '{patch_info_csv}' with {len(patch_info_df)} entries.")

if __name__ == "__main__":
    """
    Example usage:
    python generate_dataset.py
    
    This will scan the 'reference/' folder for .mp4 files,
    create a 'patches/' folder with extracted patches,
    and produce 'video_info.csv' & 'patch_info.csv'.
    """
    now = datetime.now()
    date_today = now.strftime('%Y-%m-%d')
    folder_name = f"{now.hour:02d}{now.minute:02d}"  # e.g., "1427"
    JOD_CSV = f'{VRR_Plot_HPC}/data-500_1500_2000kbps'

    PATCHSIZE = 64
    print(f'date_today {date_today}')
    for scene in scenes:
        main(
            scene,
            reference_dir=f"{VRRMP4}/uploaded/reference/{scene}",
            patches_output_dir=f"{VRRML_DATA}/ML_regression/{date_today}/{folder_name}/patches",
            video_info_csv=f"{VRRML_DATA}/ML_regression/{date_today}/{folder_name}/video_info.csv",
            patch_info_csv=f"{VRRML_DATA}/ML_regression/{date_today}/{folder_name}/patch_info.csv",
            patch_size=PATCHSIZE
        )
