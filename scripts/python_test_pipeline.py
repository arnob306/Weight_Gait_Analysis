from dotenv import load_dotenv
load_dotenv()

import os
import json
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Upload the video file
video_path = "All RGB Videos/Instance_12.0.mp4"
uploaded_file = client.files.upload(file=video_path)

prompt = (
    'You are shown a video of a person walking. Classify their gait into exactly '
    'one of the following three categories: 0 = normal walking, 1 = limping '
    '(one leg affected), 2 = shuffling (both legs affected). Respond only with '
    'valid JSON in this exact format: {"weight_class": <0, 1, or 2>, '
    '"reasoning": "<one sentence explaining what you observed>"}'
)

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=[uploaded_file, prompt],
)

print("Raw response:")
print(response.text)

# Basic JSON parsing check
try:
    parsed = json.loads(response.text.strip().strip("```json").strip("```"))
    print("\nParsed successfully:")
    print(parsed)
except json.JSONDecodeError:
    print("\n⚠️ Parse failed — response wasn't clean JSON. This is exactly the kind of thing your Methodology's parse-failure logging needs to catch.")