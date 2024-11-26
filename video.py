# Import the InferencePipeline object
from inference import InferencePipeline
# Import the built in render_boxes sink for visualizing results
from inference.core.interfaces.stream.sinks import render_boxes
import numpy as np
from arduino import send_command
import cv2
import os
from inference_sdk import InferenceHTTPClient

width = 1920
height = 1080

theta = 1020
phi = 1135

K1 = np.array([[597.6243414991088, 0.0, 922.3496765671716], [0.0, 597.5725101773556, 537.2817160922365], [0.0, 0.0, 1.0]])
R1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
T1 = np.array([[0.0], [0.0], [0.0]])
P1 = K1 @ np.hstack((R1, T1.reshape(3, 1)))

K2 = np.array([[589.1494666783758, 0.0, 922.8002910248949], [0.0, 589.8465043612913, 540.8057599235476], [0.0, 0.0, 1.0]])
R2 = np.array([[0.9999929809565761, -0.0008715335404560295, -0.0036439630718195534], [0.0008895545156719242, 0.9999873691459623, 0.004946740472323169], [0.0036396057952163044, -0.0049499472547424725, 0.9999811254677913]])
T2 = np.array([[-0.06010748017172602], [-0.000491575908121739], [-0.002582041475289312]])
P2 = K2 @ np.hstack((R2, T2.reshape(3, 1)))

# make sure to run 'inference server start' in the terminal before running this script
client = client = InferenceHTTPClient(
  api_url="http://localhost:9001",
  api_key=os.environ["ROBOFLOW_API_KEY"],
)

def sub_sink(predictions, frame):
  image = frame.image
  coords = []
  for prediction in predictions['predictions']:
    w = prediction['width']
    h = prediction['height']
    x = prediction['x'] - w // 2
    y = prediction['y'] - h // 2
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    result = client.infer(image[y:y+h, x:x+w], model_id="goose-m0dls/1")
    for prediction in result['predictions']:
      coords.append((prediction['x'] + x, prediction['y'] + y - prediction['height'] / 4))
  return coords
      
def report_prediction(predictions, frame):
  global theta
  global phi
  image = frame.image
  if len(predictions['predictions']) > 0:
    cam1_predictions = []
    cam2_predictions = []
    for coord in sub_sink(predictions, frame):
      x, y = coord
      if coord[0] < width:
        cam1_predictions.append({'x': x, 'y': y})
      else:
        cam2_predictions.append({'x': x, 'y': y})
    closest_pair = None
    closest_distance = np.Infinity
    for prediction1 in cam1_predictions:
      for prediction2 in cam2_predictions:
        x1 = prediction1['x']
        y1 = prediction1['y']
        x2 = prediction2['x'] - width
        y2 = prediction2['y']
        dx = x1 - x2
        dy = y1 - y2
        distance = dx*dx + dy*dy
        if distance < closest_distance:
          closest_distance = distance
          closest_pair = (prediction1, prediction2)
    if closest_pair:
      cv2.circle(image, (int(closest_pair[0]['x']), int(closest_pair[0]['y'])), 5, (0, 0, 255), -1)
      cv2.circle(image, (int(closest_pair[1]['x']), int(closest_pair[1]['y'])), 5, (0, 0, 255), -1)
      print(closest_pair)
      point = cv2.triangulatePoints(P1, P2, (closest_pair[0]['x'], closest_pair[0]['y']), (closest_pair[1]['x']-1920, closest_pair[1]['y']))
      laser_vec_desired = [point[0][0] - 0.3, point[1][0] - 0.381, point[2][0] + 0.0034366]
      magnitude = np.linalg.norm(laser_vec_desired)
      laser_vec_desired /= magnitude
      mirror_normal = (laser_vec_desired - np.array([0.99930814, 0.0277263, 0.02478897]))
      mirror_normal /= np.linalg.norm(mirror_normal)
      mirror_normal[0] = -mirror_normal[0]
      print(mirror_normal)
      theta = np.arctan2(mirror_normal[0], mirror_normal[2])
      phi = np.arcsin(mirror_normal[1])
      
      theta = int(1020 + theta * 1100 / (np.pi/2)) # converting to servo microseconds
      phi = int(1135 + phi * 2000 / np.pi) # converting to servo microseconds
      send_command(f"{theta} {phi} 1")
      del cam1_predictions
      del cam2_predictions
    else:
      send_command(f"{theta} {phi} 0")
  else:
    send_command(f"{theta} {phi} 0")
  
  image_small = cv2.resize(image, (int(width), int(height/2)))
  cv2.imshow("Inference", image_small)
  cv2.waitKey(1)
  

# initialize a pipeline object
pipeline = InferencePipeline.init(
  model_id="geese-detector/2", # Roboflow model to use
  video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
  on_prediction=report_prediction, # Function to run after each prediction
  video_source_properties={
    "frame_width": width*2,
    "frame_height": height,
    "fps": 60.0,
  },
)
pipeline.start()
pipeline.join()