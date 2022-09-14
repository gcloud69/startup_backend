from dataclasses import dataclass
import pifu
import mp
import cv2
import base64
import numpy as np
from flask import json
from flask import Flask
from flask import request
from flask_cors import CORS

app = Flask(__name__)

checkpoint_path = '/home/gcloud_cloud69/pifuhd/checkpoints/pifuhd.pt'
pifu_model = pifu.Pifu3DMGenerator(checkpoint_path=checkpoint_path)
pose_model = mp.MediaPipeDetector()

@app.route("/pose", methods=['POST'])
def pose():
    content_type = request.headers.get('Content-Type')
    data =  json.loads(request.data)    
    image = cv2.imdecode(np.fromstring(base64.b64decode(data['front']), dtype=np.uint8), flags=1)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results, open_pose_keypoints = pose_model.get_pose(rgb_image)
    save_path = 'output/result_output.obj'
    pifu_model.recon(image, open_pose_keypoints, save_path)

    with open(save_path, 'r') as model_file:
        return model_file.read()

if __name__ == "__main__":
    CORS(app)
    app.run(host="0.0.0.0",  debug=False, port=5000)