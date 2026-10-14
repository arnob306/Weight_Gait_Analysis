from representations import render_overlay_video

INSTANCE = 12
OUTPUT_PATH = f"overlay_Instance_{INSTANCE}.0.mp4"

render_overlay_video(INSTANCE, OUTPUT_PATH)
print(f"Saved overlay video to {OUTPUT_PATH}")
