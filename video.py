# Import the InferencePipeline object
from inference import InferencePipeline
# Import the built in render_boxes sink for visualizing results
from inference.core.interfaces.stream.sinks import render_boxes
# from laser import compute_aim_angles
# from arduino import send_command

scale = 0.5
resolution = int(1080 * scale)
width = resolution * 16 / 9


def report_prediction(predictions, frame):
  if len(predictions['predictions']) > 0:
    cam1_predictions = []
    cam2_predictions = []
    for prediction in predictions['predictions']:
      x = prediction['x']
      y = prediction['y']
      w = prediction['width']
      h = prediction['height']
      confidence = prediction['confidence']
      if x < width:
        cam1_predictions.append(prediction)
      else:
        cam2_predictions.append(prediction)
      # print(f"Prediction: x={x}, y={y}, w={w}, h={h}, confidence={round(confidence,2)}", end="\t")
    # print()
    # print(f"Cam1: {cam1_predictions}")
    # print(f"Cam2: {cam2_predictions}")
    closest_pair = None
    closest_distance = 1000000
    for prediction1 in cam1_predictions:
      for prediction2 in cam2_predictions:
        x1 = prediction1['x']
        y1 = prediction1['y']
        x2 = prediction2['x'] - width
        y2 = prediction2['y']
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if distance < closest_distance:
          closest_distance = distance
          closest_pair = (prediction1, prediction2)
    if closest_pair:
      print(f"Closest pair: {closest_pair}")
      
  # print(predictions)
  render_boxes(predictions, frame)

# initialize a pipeline object
pipeline = InferencePipeline.init(
  # model_id="goose-m0dls/2", # Roboflow model to use
  model_id="pingpong-2/1", # Roboflow model to use
  video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
  on_prediction=report_prediction, # Function to run after each prediction
  video_source_properties={
    "frame_width": width*2,
    "frame_height": resolution,
    "fps": 30.0,
  },
)
pipeline.start()
pipeline.join()