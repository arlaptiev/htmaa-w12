import tensorflow as tf
import tensorflow_hub as hub
# from tensorflow_docs.vis import embed
import numpy as np
import cv2
import time
# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display
import subprocess
from visualization import *

model_name = "movenet_lightning"
model_name = "movenet_thunder"
model_name = "multipose"
# model_name = "movenet_lightning_f16 tflite"
# model_name = "movenet_lightning_int8 tflite"

# if "tflite" in model_name:
#   if "movenet_lightning_f16" in model_name:
#     !wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite
#     input_size = 192
#   elif "movenet_thunder_f16" in model_name:
#     !wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite
#     input_size = 256
#   elif "movenet_lightning_int8" in model_name:
#     !wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite
#     input_size = 192
#   elif "movenet_thunder_int8" in model_name:
#     !wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite
#     input_size = 256
#   else:
#     raise ValueError("Unsupported model name: %s" % model_name)

if "tflite" in model_name:
  if "movenet_lightning_f16" in model_name:
      subprocess.run(['wget', '-q', '-O', 'model.tflite', 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite'])
      input_size = 192
  elif "movenet_thunder_f16" in model_name:
      subprocess.run(['wget', '-q', '-O', 'model.tflite', 'https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite'])
      input_size = 256
  elif "movenet_lightning_int8" in model_name:
      subprocess.run(['wget', '-q', '-O', 'model.tflite', 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite'])
      input_size = 192
  elif "movenet_thunder_int8" in model_name:
      subprocess.run(['wget', '-q', '-O', 'model.tflite', 'https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite'])
      input_size = 256
  else:
      raise ValueError("Unsupported model name: %s" % model_name)

  # Initialize the TFLite interpreter
  interpreter = tf.lite.Interpreter(model_path="model.tflite")
  interpreter.allocate_tensors()

  #### METAL ####
  # interpreter = tf.lite.Interpreter(model_path="model.tflite", experimental_delegates=[tf.lite.experimental.load_delegate('libtensorflow_metal_delegate.dylib')])
  # interpreter.allocate_tensors()
  #### METAL ####

  def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

else:
  if "movenet_lightning" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    # module = tf.keras.utils.get_file("saved_model", "https://tfhub.dev/google/movenet/singlepose/lightning/4")
    input_size = 192
  elif "movenet_thunder" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    # module = tf.keras.utils.get_file("saved_model", "https://tfhub.dev/google/movenet/singlepose/thunder/4")
    input_size = 256
  elif "multipose" in model_name:
     module =  hub.load("https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1")
     input_size = 256
  else:
    raise ValueError("Unsupported model name: %s" % model_name)

  def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores
  
# Load the input image.

image_path = 'squat.jpeg'

def track_pose(img, mirror = True,verbose=False, multipose = False):
  image = tf.convert_to_tensor(img, dtype=tf.uint8)

  # Resize and pad the image to keep the aspect ratio and fit the expected size.
  input_image = tf.expand_dims(image, axis=0)
  input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

  # Run model inference.
  frame_start_time = time.time()
  keypoints_with_scores = movenet(input_image)
  if verbose: print("inference:", time.time() - frame_start_time)

  # Visualize the predictions with image.
  display_image = tf.expand_dims(image, axis=0)
  display_image = tf.cast(tf.image.resize_with_pad(
      display_image, 1280, 1280), dtype=tf.int32)
  # frame_start_time = time.time()
  # output_overlay = draw_prediction_on_image(
  #     np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)
  # print("draw:", time.time() - frame_start_time)
  # output_overlay_np = tf.image.convert_image_dtype(output_overlay, dtype=tf.uint8).numpy()

  if multipose:
    # print("keypointssss", keypoints_with_scores)
    img2, result = multi_visualization(img, keypoints_with_scores, mirror=mirror)
    return img2, result
  else:
    # print("keypointssss", keypoints_with_scores)
    img2 = quick_visualization(img, keypoints_with_scores, mirror=mirror)
    return img2, keypoints_with_scores[0][0]
     
 
   

if __name__ == "__main__":
  img = cv2.imread(image_path)
  img2 = track_pose(img)
  cv2.imshow("our overlay", img2)
  cv2.waitKey(0)
  cv2.destroyAllWindows()