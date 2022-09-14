import cv2
import numpy as np
import math

import mediapipe as mp
import PIL
from IPython.display import Image, display
import cv2 
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
PL = mp_pose.PoseLandmark



def calculate_perimeter(a,b):
  return math.pi * ( 3*(a+b) - math.sqrt( (3*a + b) * (a + 3*b) ) )


class LandmarkGetter:
  def __init__(self, landmarks, width, height):
    self._landmarks = landmarks
    self._width = width
    self._height = height

  def get(self, name):
    return (self._landmarks[name].x *   self._width), (self._landmarks[name].y *   self._height)


def distance(x1, y1, x2, y2):
  return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def run_inference(front, side):
  with mp_pose.Pose(
      static_image_mode=True,
      model_complexity=1,
      enable_segmentation=True,
      min_detection_confidence=0.5) as pose:
    image_height, image_width, _ = front.shape
    front_results = pose.process(cv2.cvtColor(front, cv2.COLOR_BGR2RGB))
    image_height, image_width, _ = side.shape
    side_results = pose.process(cv2.cvtColor(side, cv2.COLOR_BGR2RGB))
    return front_results, side_results


def draw_segmentation(image, results):
  annotated_image = image.copy()
  # Draw segmentation on the image.
  # To improve segmentation around boundaries, consider applying a joint
  # bilateral filter to "results.segmentation_mask" with "image".
  condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
  bg_image = np.zeros(image.shape, dtype=np.uint8)
  bg_image[:] = (255,255,255)
  annotated_image = np.where(condition, annotated_image, bg_image)
  # Draw pose landmarks on the image.
  mp_drawing.draw_landmarks(
      annotated_image,
      results.pose_landmarks,
      mp_pose.POSE_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
  
  return annotated_image

def raster_scan(mask, start_x, start_y, direction=1):
  points = []
  for shift in range(0,20,5):
    start_x = int(start_x)
    start_y = int(start_y) + shift
    while(mask[start_y][start_x] == 1):
      start_x += direction
    points.append((start_x, start_y))
  return points
    

def get_height_from_seg(results, width, height):
  mask = np.where(results.segmentation_mask>0.8, 1, 0)
  getter = LandmarkGetter(results.pose_landmarks.landmark, width, height)
  nose = getter.get(PL.NOSE)
  lheel = getter.get(PL.LEFT_HEEL)
  # cv2.circle(annotated_image, (j, dy), 4, color=(255,0,0))
  px = lheel[1] - nose[1] + 10
  return px


def get_waist(front_res, side_res, width, height, ratio):
  side_mask = np.where(side_res.segmentation_mask>0.8, 1, 0)
  side_getter = LandmarkGetter(side_res.pose_landmarks.landmark, width, height)

  rhip = side_getter.get(PL.RIGHT_HIP)
  lhip = side_getter.get(PL.LEFT_HIP)
  rpoints = raster_scan(side_mask, rhip[0], rhip[1])
  lpoints = raster_scan(side_mask, lhip[0], lhip[1], -1)

  side_hip = rpoints[0][0] - lpoints[0]

  front_mask = np.where(front_res.segmentation_mask>0.8, 1, 0)
  front_getter = LandmarkGetter(front_res.pose_landmarks.landmark, width, height)
  lhip = front_getter.get(PL.LEFT_HIP)
  rhip = front_getter.get(PL.RIGHT_HIP)

  hip = lhip[0] - rhip[0]

  box = (2*hip + 2*side_hip) * ratio
  ellipse = calculate_perimeter(hip/2, side_hip/2) * ratio




def get_chest(front_res, side_res, width, height, ratio):
  side_mask = np.where(side_res.segmentation_mask>0.8, 1, 0)
  side_getter = LandmarkGetter(side_res.pose_landmarks.landmark, width, height)
  rshoulder = side_getter.get(PL.RIGHT_SHOULDER)
  rpoints = raster_scan(side_mask, rshoulder[0], rshoulder[1])
  lpoints = raster_scan(side_mask, rshoulder[0], rshoulder[1], -1)

  side_chest = rpoints[0][0] - rshoulder[0]
  side_chest_full = rpoints[0][0] - lpoints[0][0]

  front_mask = np.where(front_res.segmentation_mask>0.8, 1, 0)
  front_getter = LandmarkGetter(front_res.pose_landmarks.landmark, width, height)
  lshoulder = front_getter.get(PL.LEFT_SHOULDER)
  rshoulder = front_getter.get(PL.RIGHT_SHOULDER)

  chest = lshoulder[0] - rshoulder[0]

  box = (2*chest + 2*side_chest_full) * ratio
  ellipse = calculate_perimeter(chest/2, side_chest) * ratio

  return box, ellipse

def get_measurments():
  front, side = cv2.imread('front.jpeg'), cv2.imread('side.jpeg')
  height, width, _ = front.shape
  front_res, side_res = run_inference(front, side)
  segmentation = draw_segmentation(front, front_res)
  # cv2.circle(annotated_image, (j, dy), 4, color=(255,0,0))
  height_px = get_height_from_seg(front_res, width, height)
  ratio = 170/height_px

  box, ellipse = get_chest(front_res, side_res, width, height, ratio)
  print(get_waist(fton_res, side_res, width, height, ratio))
  print(box, ellipse, (box+ellipse)/2)

get_measurments()
