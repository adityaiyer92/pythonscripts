"""
Aditya Iyer
Displays a random image from the dataset's target directory and class combination.
dataset_name: Name of the dataset
target_dir: train or test
target_class: class name
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os

# Let's Viusalize the Images

def view_random_iamge(dataset_name, target_dir, target_class):
  """
  Displays a random image from the dataset's target directory and class combination.
  dataset_name: Name of the dataset
  target_dir: train or test
  target_class: class name
  """
  # Setup the target directory
  target_folder = dataset_name + "/" + target_dir + "/" + target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)
  print(random_image)

  # Read in the image and plot it
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  # Show the shape of the image
  print(f"Image shape: {img.shape}")

  return img
