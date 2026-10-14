from representations import render_skeleton_video

INSTANCE = 12
OUTPUT_PATH = f"skeleton_Instance_{INSTANCE}.0.mp4"

render_skeleton_video(INSTANCE, OUTPUT_PATH)
print(f"Saved skeleton video to {OUTPUT_PATH}")
