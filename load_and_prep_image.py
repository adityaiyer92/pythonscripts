"""
Aditya Iyer
"""
import tensorflow as tf

def load_and_prep_image(filename, img_shape= 224):
  """
  Reads an image from the filename, turns it into a tensor and reshapes it to
  (img_shape, img_shape, color_channel).
  """
  # Read in the image
  img = tf.io.read_file(filename)

  # Decode the read file into a tensor
  img = tf.image.decode_image(img)

  # Resize image
  img = tf.image.resize(img, size= [img_shape, img_shape])

  # Rescale the image to get values to between 0 and 1
  img = img/255.

  return img