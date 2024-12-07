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

# make sure to run the docker server first
client = InferenceHTTPClient(
  api_url="http://localhost:9001",
  api_key=os.environ["ROBOFLOW_API_KEY"],
)

def DLT(P1, P2, point1, point2):

  A = [point1[1]*P1[2,:] - P1[1,:],
    P1[0,:] - point1[0]*P1[2,:],
    point2[1]*P2[2,:] - P2[1,:],
    P2[0,:] - point2[0]*P2[2,:]
  ]
  A = np.array(A).reshape((4,4))

  B = A.transpose() @ A
  U, s, Vh = np.linalg.svd(B, full_matrices = False)

  return Vh[3,0:3]/Vh[3,3]

def calculate_microseconds(point):
  laser_vec_desired = point + np.array([-0.03, -0.0381, 0.0034366])
  magnitude = np.linalg.norm(laser_vec_desired)
  laser_vec_desired /= magnitude
  mirror_normal = laser_vec_desired - np.array([0.99930814, 0.0277263, 0.02478897])
  mirror_normal /= np.linalg.norm(mirror_normal)
  mirror_normal[0] = -mirror_normal[0]
  theta = np.arctan2(mirror_normal[0], mirror_normal[2])
  phi = np.arcsin(mirror_normal[1])
  
  theta = int(1020 + theta * 1100 / (np.pi/2)) # converting to servo microseconds
  phi = int(1135 + phi * 2000 / np.pi) # converting to servo microseconds
  
  return theta, phi

if __name__ == "__main__":
  camera = cv2.VideoCapture(0)
  camera.set(cv2.CAP_PROP_FRAME_WIDTH, width*2)
  camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  coords1 = None
  coords2 = None
  p3ds = None
  while True:
    coords1 = []
    coords2 = []
    p3ds = []
    ret, image = camera.read()
    result = client.infer(image, model_id="geese-detector/2")
    for prediction in result['predictions']:
      w = prediction['width']
      h = prediction['height']
      x = prediction['x'] - w // 2
      y = prediction['y'] - h // 2
      x = int(x)
      y = int(y)
      w = int(w)
      h = int(h)
      cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
      result = client.infer(image[y:y+h, x:x+w], model_id="goose-m0dls/1")
      for prediction in result['predictions']:
        x = int(prediction['x'] + x)
        y = int(prediction['y'] + y - prediction['height'] / 4)
        if x > width:
          coords2.append(np.array([[x-width], [y]]))
        else:
          coords1.append(np.array([[x], [y]]))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    for p1 in coords1:
      min_dis = np.inf
      min_point = None
      for p2 in coords2:
        dis = np.linalg.norm(p1 - p2)
        if dis < 200 and dis < min_dis:
          min_dis = dis
          min_point = p2
      if min_point is not None:
        p3d = DLT(P1, P2, p1, min_point)
        cv2.line(image, (int(p1[0]), int(p1[1])), (int(min_point[0]+width), int(min_point[1])), (0, 0, 255), 2)
        p3ds.append(p3d)
    min_dis = np.inf
    min_point = None
    for point in p3ds:
      dis = np.linalg.norm(point - np.array([[0], [0], [0]]))
      if dis < min_dis:
        min_dis = dis
        min_point = point
    if min_point is not None: 
      theta, phi = calculate_microseconds(min_point)
      send_command(f"{theta} {phi} 1")
    else:
      send_command(f"{theta} {phi} 0")
    image_small = cv2.resize(image, fx=0.25, fy=0.25, dsize=(0, 0))
    cv2.imshow("image", image_small)
    k = cv2.waitKey(1)
    if k == ord("q"):
      break
  camera.release()
  cv2.destroyAllWindows()