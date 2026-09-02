import cv2
cap = cv2.VideoCapture("All RGB Videos/Instance_12.0.mp4")
print("Frame count:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
cap.release()

import pandas as pd
df = pd.read_csv("All Joints/Absolute_Data.csv", header=None)
instance = df[df[0] == 12]
print(instance.iloc[:, :6])
print(f"Row count: {len(instance)}")

import cv2, pandas as pd, os

df = pd.read_csv("All Joints/Absolute_Data.csv", header=None)
mismatches = []

for i in range(1920):
    video_path = f"All RGB Videos/Instance_{i}.0.mp4"
    if not os.path.exists(video_path):
        mismatches.append((i, "missing video"))
        continue
    cap = cv2.VideoCapture(video_path)
    vframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    crows = len(df[df[0] == i])
    if vframes != crows:
        mismatches.append((i, f"video={vframes}, csv={crows}"))

print(f"{len(mismatches)} mismatches out of 1920")
for m in mismatches[:20]:
    print(m)