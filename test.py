from inference import InferencePipeline
import inference.core.interfaces.camera.entities
from inference.core.interfaces.stream.sinks import render_boxes
import cv2
import os
from inference_sdk import InferenceHTTPClient

client = client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)

def sink(predictions: dict, frame: inference.core.interfaces.camera.entities.VideoFrame):
  image = frame.image
  goose_images = []
  for prediction in predictions['predictions']:
    w = prediction['width']
    h = prediction['height']
    x = prediction['x'] - w // 2
    y = prediction['y'] - h // 2
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    goose_images.append(((x,y),image[y:y+h, x:x+w]))
  results = []
  for goose_image in goose_images:
    (x,y),goose_image = goose_image
    result = client.infer(goose_image, model_id="goose-m0dls/1")
    for prediction in result['predictions']:
      prediction['x'] += x
      prediction['y'] += y
      w = prediction['width']
      h = prediction['height']
      x = prediction['x'] - w // 2
      y = prediction['y'] - h // 2
      x = int(x)
      y = int(y)
      w = int(w)
      h = int(h)
      cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    results.append(result)
  for result in results:
    predictions = result['predictions']
    for prediction in predictions:
      x = prediction['x']
      y = prediction['y']
      y -= prediction['height'] / 4
      x = int(x)
      y = int(y)
      cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
  cv2.imshow("Inference", image)
  cv2.waitKey(1)
    

# initialize a pipeline object
pipeline = InferencePipeline.init(
  model_id="geese-detector/2", # Roboflow model to use
  video_reference="geese.mp4", # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
  on_prediction=sink, # Function to run after each prediction
)
pipeline.start()
pipeline.join()