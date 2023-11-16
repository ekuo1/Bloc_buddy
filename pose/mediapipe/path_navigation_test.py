from pathlib import Path

name = "blue_overhang"
script_path = Path(__file__).parent
print(script_path)
video_location = f"../test_examples/videos/{name}.mp4"

video_path = script_path / video_location
print(video_path.resolve())

