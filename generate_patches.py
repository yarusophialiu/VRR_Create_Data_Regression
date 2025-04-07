import os
import cv2
import random
import numpy as np
import pandas as pd
from utils import *
from datetime import datetime

# ------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------

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
        # If frame is too small, just return the center (or skip).
        start_y = max(0, h // 2 - patch_size // 2)
        start_x = max(0, w // 2 - patch_size // 2)
    else:
        start_y = random.randint(0, h - patch_size)
        start_x = random.randint(0, w - patch_size)

    patch = frame[start_y:start_y+patch_size, start_x:start_x+patch_size, :]
    return patch


def parse_video_id_from_path(video_path):
    """
    Example function to parse a 'video_id' from a path string.
    We take the 3rd-last folder name as the ID if possible.
    e.g. reference/bedroom/bedroom_path1_seg1_1/ref166_1080/refOutput.mp4
         -> "bedroom_path1_seg1_1"
    Adjust as needed for your directory structure.
    """
    parts = video_path.split(os.sep)
    if len(parts) >= 3:
        folder2 = parts[-3]  # e.g. "bedroom_path1_seg1_1"
        video_id = f"{folder2}"
    else:
        # fallback if path shorter than expected
        filename = os.path.splitext(parts[-1])[0]
        video_id = filename
    return video_id


def read_frame_velocity(frame_velocity_path, frame_number):
    """
    Example: read a text file listing 'frame velocity' lines,
    find the line matching frame_number, return float(velocity).
    """
    if not os.path.exists(frame_velocity_path):
        return None
    with open(frame_velocity_path, "r") as file:
        for line in file:
            frame, velocity = line.split()
            if int(frame) == frame_number:
                return float(velocity)
    return None  # if frame not found


# ------------------------------------------------------------------
# Reading JOD from Excel
# ------------------------------------------------------------------

def find_jod_rows(scene, sheet_name):
    """
    Read an Excel sheet 'sheet_name' (e.g., path1_seg1_1)
    from file '{scene}.xlsx' in JOD_CSV folder.
    Returns a list of dicts, each dict representing one row:
      {
        'video_id': f"{scene}_{sheet_name}",
        'bitrate': <int>,
        'jod_0': <float>,
        ...
        'jod_49': <float>
      }
    If the sheet has multiple rows (like multiple bitrates),
    we'll return one dict per row.
    """
    file_path = f"{JOD_CSV}/{scene}.xlsx"
    if not os.path.exists(file_path):
        print(f"[Warning] Excel file not found: {file_path}")
        return []

    print(f"Reading sheet '{sheet_name}' from '{file_path}' ...")
    try:
        xls = pd.ExcelFile(file_path)
        if sheet_name not in xls.sheet_names:
            print(f"[Warning] Sheet '{sheet_name}' not found in {file_path}")
            return []
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    except Exception as e:
        print(f"[Error] Could not read {sheet_name} in {file_path}: {e}")
        return []

    # 50 combos: (360,480,720,864,1080) repeated for fps(30..120 in steps of 10)
    resolutions = [i for _ in range(10) for i in [360, 480, 720, 864, 1080]]
    fps_headers = [i for i in range(30, 121, 10) for _ in range(5)]

    column_names = ["bitrate"] + [f"{res}-fps{fps}" for res, fps in zip(resolutions, fps_headers)]

    # The first row is usually a header, so skip it
    # We'll parse from row index=1 onward
    data = df.iloc[1:, :len(column_names)].reset_index(drop=True)
    data.columns = column_names

    # Clean up the 'bitrate' column
    data["bitrate"] = pd.to_numeric(data["bitrate"], errors="coerce")
    data = data.dropna(subset=["bitrate"]).copy()
    data["bitrate"] = data["bitrate"].astype(int)
    if data.empty:
        print(f"[Warning] No valid bitrate rows in sheet '{sheet_name}' of {scene}")
        return []

    row_dicts = []
    for idx in range(len(data)):
        row = data.iloc[idx]
        bitrate = row["bitrate"]
        # Extract 50 JOD values
        jod_values = row[1:].tolist()  # skip the 'bitrate' col

        # Build the dictionary
        row_dict = {
            "video_id": f"{scene}_{sheet_name}",
            "bitrate": bitrate
        }
        for j in range(len(jod_values)):
            row_dict[f"jod_{j}"] = float(jod_values[j])
        row_dicts.append(row_dict)

    return row_dicts


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(scene, reference_dir="reference", 
         patches_output_dir="patches",
         video_info_csv="video_info.csv",
         patch_info_csv="patch_info.csv",
         patch_size=128):
    """
    1) Gather all MP4 videos from reference_dir.
    2) For each video_id:
       - read actual JOD rows from Excel using find_jod_rows(scene, pathX_segY)
         and store them in video_info_rows
       - open the video, extract up to 5 patches (or however many frames),
         read velocity, store patch info in patch_info_rows
    """

    # ------------------------------------------------------------------
    # 1) Collect all .mp4 files
    # ------------------------------------------------------------------
    video_paths = gather_mp4_files(reference_dir) # len(video_paths) = 45

    # Prepare lists to store rows for CSVs
    video_info_rows = []
    patch_info_rows = []

    os.makedirs(patches_output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2) Iterate over each video file
    # ------------------------------------------------------------------
    # count = 0
    for vid_path in video_paths:
        video_id = parse_video_id_from_path(vid_path)  # e.g. "bedroom_path1_seg1_2"
        print(f"\nProcessing video_id: {video_id}")

        # e.g. "bedroom_path1_seg1_2" => "path1_seg1_2"
        if "_path" in video_id:
            path_info = "path" + video_id.split("_path", 1)[1]
        else:
            # fallback if no "_path" in name
            path_info = video_id

        # read JOD rows from Excel
        jod_rows = find_jod_rows(scene, path_info)
        video_info_rows.extend(jod_rows)

        # ------------------------------------------------------------------
        # 2a) Extract patches from the video
        # ------------------------------------------------------------------
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video: {vid_path}")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            print(f"Warning: Zero frames in video: {vid_path}")
            cap.release()
            continue

        frame_velocity_path = f"{VRR_Motion}/reference/magnitude_motion_per_frame/{scene}/{video_id}_velocity_per_frame.txt"

        frames_extracted = 0
        frame_number = 0
        while  cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # no more frames

            frame_velocity = read_frame_velocity(frame_velocity_path, frame_number)
            if frame_velocity is None:
                frame_number += 1
                continue

            patch_bgr = extract_random_patch(frame, patch_size=patch_size)

            patch_filename = f"{video_id}_frame{frame_number}.jpg"
            patch_path_full = os.path.join(patches_output_dir, patch_filename)
            # cv2.imwrite(patch_path_full, patch_bgr)

            # Add row to patch_info
            patch_info_rows.append({
                "patch_path": patch_filename,  # or patch_path_full
                "video_id": video_id,
                "velocity": frame_velocity
            })

            frames_extracted += 1
            frame_number += 1
        print(f'{frames_extracted} frames extracted for {video_id}.')

        cap.release()
        break

    # ------------------------------------------------------------------
    # 3) Save CSVs
    # ------------------------------------------------------------------

    if video_info_rows:
        video_info_df = pd.DataFrame(video_info_rows)
        video_info_df.drop_duplicates(subset=["video_id", "bitrate"], inplace=True)

        write_header = not os.path.exists(video_info_csv)
        video_info_df.to_csv(video_info_csv, mode='a', header=write_header, index=False)
        print(f"Appended to video_info: '{video_info_csv}' (+{len(video_info_df)} rows)")
    else:
        print("No new video_info to append.")


    if patch_info_rows:
        patch_info_df = pd.DataFrame(patch_info_rows)

        write_header = not os.path.exists(patch_info_csv)
        patch_info_df.to_csv(patch_info_csv, mode='a', header=write_header, index=False)
        print(f"Appended to patch_info: '{patch_info_csv}' (+{len(patch_info_df)} rows)")
    else:
        print("No new patch_info to append.")



# ------------------------------------------------------------------
# Running the Script
# ------------------------------------------------------------------

if __name__ == "__main__":
    """
    Example usage:
    python generate_dataset.py

    - Scans 'reference_dir' for .mp4 files (one video_id per .mp4).
    - Creates a 'patches/' folder with extracted patches.
    - Produces 'video_info.csv' & 'patch_info.csv'.
    """
    now = datetime.now()
    date_today = now.strftime('%Y-%m-%d')
    folder_name = f"{now.hour:02d}{now.minute:02d}"  # e.g., "1427"
    JOD_CSV = f'{VRR_Plot_HPC}/data-500_1500_2000kbps'

    PATCHSIZE = 64

    now = datetime.now()
    date_today = now.strftime('%Y-%m-%d')
    folder_name = f"{now.hour:02d}{now.minute:02d}"  # e.g., "1427"

    PATCHSIZE = 64

    for scene in scenes:
        main(
            scene,
            reference_dir=f"{VRRMP4}/uploaded/reference/{scene}",
            patches_output_dir=f"{VRRML_DATA}/ML_regression/{date_today}/{folder_name}/patches",
            video_info_csv=f"{VRRML_DATA}/ML_regression/{date_today}/{folder_name}/video_info.csv",
            patch_info_csv=f"{VRRML_DATA}/ML_regression/{date_today}/{folder_name}/patch_info.csv",
            patch_size=PATCHSIZE
        )
