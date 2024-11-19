# Import the InferencePipeline object
from inference import InferencePipeline
# Import the built in render_boxes sink for visualizing results
from inference.core.interfaces.stream.sinks import render_boxes
import numpy as np
from scipy import linalg
# from laser import compute_aim_angles
# from arduino import send_command

scale = 1
width = 1920*scale
height = 1080*scale

K1 = np.array([[575.2471989038615, 0.0, 922.7536420263076], [0.0, 575.2431210944071, 536.7506075573181], [0.0, 0.0, 1.0]])
R1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
T1 = np.array([0.0, 0.0, 0.0])
P1 = K1 @ np.hstack((R1, T1.reshape(3, 1)))

K2 = np.array([[574.749735857808, 0.0, 923.5729228372368], [0.0, 574.593809271748, 540.0674308712719], [0.0, 0.0, 1.0]])
R2 = np.array([[0.9999645033314316, -0.0009832151196831449, -0.008368115985804937], [0.0010038095237407322, 0.999996477231378, 0.002457212093844406], [0.008365670538785584, -0.0024655248655234174, 0.9999619676485572]])
T2 = np.array([-0.059356081226967713, -0.00032217644167583363, -0.00041278943555724885])
P2 = K2 @ np.hstack((R2, T2.reshape(3, 1)))

def DLT(point1, point2):
  
  A = [point1[1]*P1[2,:] - P1[1,:],
       P1[0,:] - point1[0]*P1[2,:],
       point2[1]*P2[2,:] - P2[1,:],
       P2[0,:] - point2[0]*P2[2,:]
      ]
  A = np.array(A).reshape((4,4))

  B = A.transpose() @ A
  U, s, Vh = linalg.svd(B, full_matrices = False)

  #print('Triangulated point: ')
  #print(Vh[3,0:3]/Vh[3,3])
  point = Vh[3,0:3]/Vh[3,3]
  point[0] += .03
  point[1] -= .0381
  point[2] -= 0.0034366
  return point

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
      # print(f"Closest pair: {closest_pair}")
      point = DLT((closest_pair[0]['x']/scale, 1080-closest_pair[0]['y']/scale), (closest_pair[1]['x']/scale-1920, 1080-closest_pair[1]['y']/scale))
      magnitude = (point[0]**2 + point[1]**2 + point[2]**2)**.5
      print(point, magnitude)
      
  # print(predictions)
  render_boxes(predictions, frame)

# initialize a pipeline object
pipeline = InferencePipeline.init(
  # model_id="goose-m0dls/2", # Roboflow model to use
  model_id="face-detection-mik1i/21", # Roboflow model to use
  video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
  on_prediction=report_prediction, # Function to run after each prediction
  video_source_properties={
    "frame_width": width*2,
    "frame_height": height,
    "fps": 30.0,
  },
)
pipeline.start()
pipeline.join()