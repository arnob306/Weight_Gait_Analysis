import cv2
import numpy as np

from skeleton_utils import draw_skeleton, joints_from_row, load_instance_joints

INSTANCE = 12
CSV_PATH = "All Joints/Absolute_Data.csv"
RAW_VIDEO_PATH = f"All RGB Videos/Instance_{INSTANCE}.0.mp4"
OUTPUT_PATH = f"skeleton_Instance_{INSTANCE}.0.mp4"

cap = cv2.VideoCapture(RAW_VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print(f"Raw video: {width}x{height} @ {fps:.2f} fps")

inst = load_instance_joints(CSV_PATH, INSTANCE)
print(f"Frames in CSV for instance {INSTANCE}: {len(inst)}")

writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

for _, row in inst.iterrows():
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    joints, valid = joints_from_row(row)
    draw_skeleton(canvas, joints, valid)
    writer.write(canvas)

writer.release()
print(f"Saved skeleton video to {OUTPUT_PATH}")
