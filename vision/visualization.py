import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import matplotlib
from bidict import bidict

# matplotlib.use('agg')
matplotlib.use('TkAgg')

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = bidict({
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
})

MIRRORED_KEYPOINT_DICT = bidict({
    'nose': 0,
    'right_eye': 1,
    'left_eye': 2,
    'right_ear': 3,
    'left_ear': 4,
    'right_shoulder': 5,
    'left_shoulder': 6,
    'right_elbow': 7,
    'left_elbow': 8,
    'right_wrist': 9,
    'left_wrist': 10,
    'right_hip': 11,
    'left_hip': 12,
    'right_knee': 13,
    'left_knee': 14,
    'right_ankle': 15,
    'left_ankle': 16
   
})

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

MATPLOT_COLORS_COLOR_TO_RGB = {
    'm': (255, 0, 255),
    'c': (0, 255, 255),
    'y': (255, 255, 0)
}


def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12), dpi=100)
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot

# def to_gif(images, duration):
#   """Converts image sequence (4D numpy array) to gif."""
#   imageio.mimsave('./animation.gif', images, duration=duration)
#   return embed.embed_file('./animation.gif')

def progress(value, max=100):
  return HTML("""
      <progress
          value='{value}'
          max='{max}',
          style='width: 100%'
      >
          {value}
      </progress>
  """.format(value=value, max=max))

def quick_visualization(img, keypoints, min_score = .11, arm='right', mirror = True, verbose = False):
  if mirror:
      KP_DICT = MIRRORED_KEYPOINT_DICT
  else:
      KP_DICT = KEYPOINT_DICT
  img = resize_and_pad(img, (1280, 1280))
  width, height = img.shape[:2]
  keypoints = keypoints[0][0]
  if verbose: print("keypoints:", keypoints) 
  for index, (y, x, score) in enumerate(keypoints):
      if verbose: print("x",x*width,"y:", y*height,"score:", score)
      if score > min_score:
          if KP_DICT[f'{arm}_wrist'] == index or KP_DICT[f'{arm}_elbow'] == index:
              cv2.circle(img, (int(x*width), int(y*height)), 15, (0, 255, 0), -1)
          else:
              cv2.circle(img, (int(x*width), int(y*height)), 5, (0, 0, 255), -1)
  for i1, i2 in KEYPOINT_EDGE_INDS_TO_COLOR:
      p1 = keypoints[i1]
      p2 = keypoints[i2]
      if p1[2] > min_score and p2[2] > min_score:
          cv2.line(img, (int(p1[1]*width), int(p1[0]*height)), (int(p2[1]*width), int(p2[0]*height)), MATPLOT_COLORS_COLOR_TO_RGB[KEYPOINT_EDGE_INDS_TO_COLOR[(i1, i2)]], 3)
  return img

def multi_visualization(img, keypoints, min_score = .11, mirror = True, verbose = False):
  if mirror:
    KP_DICT = MIRRORED_KEYPOINT_DICT
  else:
    KP_DICT = KEYPOINT_DICT
  img = resize_and_pad(img, (1280, 1280))
  width, height = img.shape[:2]
  keypoints = keypoints[0]
  if verbose: print("keypoints:", keypoints) 

  result = []
  for kps in keypoints:
    res_dict = {}
    kps = kps[:51]
    reorganized = [(kps[i], kps[i + 1], kps[i + 2]) for i in range(0, len(kps), 3)]
    # print("reorganized", reorganized)
    for index, (y, x, score) in enumerate(reorganized):
      if verbose: print("x",x*width,"y:", y*height,"score:", score)
      if score > min_score:
        res_dict[KP_DICT.inv[index]] = (y*height, x*width, score)
        if KP_DICT['right_shoulder'] == index or KP_DICT['right_hip'] == index or KP_DICT['left_shoulder'] == index or KP_DICT['left_hip'] == index:
          cv2.circle(img, (int(x*width), int(y*height)), 15, (0, 255, 0), -1)
        else:
          cv2.circle(img, (int(x*width), int(y*height)), 5, (0, 0, 0), -1)

    if 'right_shoulder' in res_dict and 'right_hip' in res_dict and 'left_shoulder' in res_dict and 'left_hip' in res_dict:
      result.append(res_dict)
      tx = (res_dict['left_hip'][1] + res_dict['right_hip'][1] + res_dict['left_shoulder'][1] + res_dict['right_shoulder'][1])/4
      ty = (res_dict['left_hip'][0] + res_dict['right_hip'][0] + res_dict['left_shoulder'][0] + res_dict['right_shoulder'][0])/4
      cv2.circle(img, (int(tx), int(ty)), 15, (0, 0, 255), -1)
  
  return img, result
   

def resize_and_pad(image, target_size):
    # Get the original image size
    original_height, original_width, _ = image.shape

    # Calculate the aspect ratio of the original image
    aspect_ratio = original_width / float(original_height)

    # Calculate the target width and height based on the target size
    target_width, target_height = target_size

    # Calculate the scaling factors for width and height
    width_scale = target_width / original_width
    height_scale = target_height / original_height

    # Use the minimum scaling factor to preserve the aspect ratio
    scale = min(width_scale, height_scale)

    # Resize the image with the calculated scale
    resized_image = cv2.resize(image, None, fx=scale, fy=scale)

    # Calculate the padding needed to achieve the target size
    padding_x = int((target_width - resized_image.shape[1]) / 2)
    padding_y = int((target_height - resized_image.shape[0]) / 2)

    # Create a canvas with the target size and fill it with padding color (e.g., white)
    padded_image = np.full((target_height, target_width, 3), 255, dtype=np.uint8)

    # Place the resized image in the center of the canvas
    padded_image[padding_y:padding_y + resized_image.shape[0], padding_x:padding_x + resized_image.shape[1]] = resized_image

    return padded_image