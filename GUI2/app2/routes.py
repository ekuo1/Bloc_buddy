from app2 import app2
import time, sys
from flask import request, jsonify, redirect, url_for
from pathlib import Path, PureWindowsPath

# Find path of file name relative to the running script
script_path = Path(__file__).parent # identifies path where the script is

root_path = script_path / "../../"
root_path = str(root_path.resolve())
sys.path.append(root_path)

from video_processing import setup, process_video
compiled_model_ir, mp_drawing, mp_drawing_styles, mp_pose, pose = setup()

@app2.route('/')
@app2.route('/index')
def index():
    return "Hello, World!"

@app2.route('/receive_data', methods=['GET'])
def receive_data():
    received_data = request.args  # Retrieve the data from the request's query parameters
    name = received_data.get('name')
    climb_id = received_data.get('climb_id')
    print(f"Received data: name = {name}, id = {climb_id}")
    # append GUI folder to the path
    gui_path = script_path / "../../GUI"
    gui_path = str(gui_path.resolve())
    sys.path.append(gui_path)

    # Import the flask app and its modules
    from app import app
    with app.app_context():
        process_video(name, compiled_model_ir, mp_drawing, mp_drawing_styles, mp_pose, pose, climb_id=climb_id, app_context=app.app_context())
    return redirect(url_for('index'))