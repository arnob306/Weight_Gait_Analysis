import cv2

from skeleton_utils import draw_skeleton, joints_from_row, load_instance_joints

INSTANCE = 12
CSV_PATH = "All Joints/Absolute_Data.csv"
RAW_VIDEO_PATH = f"All RGB Videos/Instance_{INSTANCE}.0.mp4"
OUTPUT_PATH = f"overlay_Instance_{INSTANCE}.0.mp4"

cap = cv2.VideoCapture(RAW_VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Raw video: {width}x{height} @ {fps:.2f} fps")

inst = load_instance_joints(CSV_PATH, INSTANCE)
print(f"Frames in CSV for instance {INSTANCE}: {len(inst)}")

writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

for _, row in inst.iterrows():
    ok, frame = cap.read()
    if not ok:
        break
    joints, valid = joints_from_row(row)
    draw_skeleton(frame, joints, valid)
    writer.write(frame)

cap.release()
writer.release()
print(f"Saved overlay video to {OUTPUT_PATH}")
