from representations import render_montage_image

INSTANCE = 12
OUTPUT_PATH = f"montage_Instance_{INSTANCE}.0.png"

render_montage_image(INSTANCE, OUTPUT_PATH)
print(f"Saved gait-cycle montage to {OUTPUT_PATH}")
