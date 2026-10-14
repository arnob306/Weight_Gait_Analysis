from dotenv import load_dotenv
load_dotenv()
import os
from google import genai
from harness import classify_clip

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

prompt = (
    'You are shown a video of a person walking. Classify their gait into exactly '
    'one of the following three categories: 0 = normal walking, 1 = limping '
    '(one leg affected), 2 = shuffling (both legs affected). Respond only with '
    'valid JSON in this exact format: {"weight_class": <0, 1, or 2>, '
    '"reasoning": "<one sentence explaining what you observed>"}'
)

result = classify_clip(client, "overlay_Instance_12.0.mp4", prompt)

print("Raw response:", result["raw_response"])
print("Success:", result["success"])
if result["success"]:
    print("Parsed:", result["parsed"])
else:
    print("Parse error:", result["error"])