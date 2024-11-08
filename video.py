# Import the InferencePipeline object
from inference import InferencePipeline
# Import the built in render_boxes sink for visualizing results
from inference.core.interfaces.stream.sinks import render_boxes
from laser import compute_aim_angles
from arduino import send_command

def report_prediction(predictions, frame):
  print(predictions)
  render_boxes(predictions, frame)

# initialize a pipeline object
pipeline = InferencePipeline.init(
  model_id="goose-m0dls/2", # Roboflow model to use
  video_reference=2, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
  on_prediction=report_prediction, # Function to run after each prediction
)
pipeline.start()
pipeline.join()