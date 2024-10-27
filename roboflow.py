from io import BytesIO

import requests
import supervision as sv
from inference import get_model
from PIL import Image
from PIL.ImageFile import ImageFile


def load_image_from_url(url: str) -> ImageFile:
  response = requests.get(url)
  response.raise_for_status()  # check if the request was successful
  image = Image.open(BytesIO(response.content))
  return image

def get_head_locations(image: ImageFile) -> sv.Detections:
  model = get_model(model_id="goose-m0dls/1")
  results = model.infer(image)
  detections = sv.Detections.from_inference(results)
  # return dictionary of boxes with confidence greater than 0.5

def main():
  # load the image from an url
  image = Image.open("C:/Users/brian/OneDrive/Desktop/Projects/ENGR 100/goose-annihilator/geese/goose3.jpg")

  # load a pre-trained yolov8n model
  model = get_model(model_id="goose-m0dls/1")

  # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
  results = model.infer(image)[0]

  # load the results into the supervision Detections api
  detections = sv.Detections.from_inference(results)

  # create supervision annotators
  bounding_box_annotator = sv.BoxAnnotator()
  label_annotator = sv.LabelAnnotator()

  # annotate the image with our inference results
  annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
  annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

  # display the image
  sv.plot_image(annotated_image)
  
if(__name__ == "__main__"):
  main()