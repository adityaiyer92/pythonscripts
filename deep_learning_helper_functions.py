"""
Aditya Iyer, 2025
Helper functions to be used in deep learning projects.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# Function to extract contents of a zip file
def extract_zip(zip_path):
    """
    Extracts the contents of a zip file to the current directory.
    :param zip_path: Path to the zip file.
    returns: None
    """
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall()
    zip_ref.close()

# Function to programmatically get class names from a directory
def get_class_names(directory):
    """
    Returns the names of the subdirectories in the given directory.
    :param directory: Path to the directory.
    returns: List of class names.
    """
    return np.array(sorted(os.listdir(directory)))

# Function to display a random image from a directory
def display_random_image(directory, class_names):
    """
    Displays a random image from the given directory.
    :param directory: Path to the directory.
    returns: None
    """
    
    class_name = np.random.choice(class_names)
    image_path = os.path.join(directory, class_name)
    image_name = np.random.choice(os.listdir(image_path))
    image = plt.imread(os.path.join(image_path, image_name))
    plt.imshow(image)
    plt.axis('off')
    plt.title(class_name)
    plt.show()

# Function to display a random batch of images from a directory
def display_random_batch(directory, class_names, batch_size):
    """
    Displays a random batch of images from the given directory.
    :param directory: Path to the directory.
    :param batch_size: Number of images to display.
    returns: None
    """
    
    class_name = np.random.choice(class_names)
    image_path = os.path.join(directory, class_name)
    image_names = np.random.choice(os.listdir(image_path), batch_size)
    
    plt.figure(figsize=(batch_size*2, 2))
    for i in range(batch_size):
        image = plt.imread(os.path.join(image_path, image_names[i]))
        plt.subplot(1, batch_size, i+1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(class_name)
    plt.show()

# Function to plot the loss curves from model history
def plot_loss_curves(history):
    """
    Plots the loss curves from the model history.
    :param history: Model history.
    returns: None
    """
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    plt.figure(figsize= (12, 5))
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label= 'training_loss')
    plt.plot(epochs, val_loss, label= 'validation_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend();

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label= 'training_accuracy')
    plt.plot(epochs, val_accuracy, label= 'validation_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

# Function to load and preprocess single image
def load_and_prep_image(filename, img_shape=224):
    """
    Reads an image from a file and preprocesses it.
    :param filename: Path to the image file.
    :param img_shape: Desired image shape.
    returns: Preprocessed image.
    """
    
    # Read in the image
    img = tf.io.read_file(filename)

    # Decode the image
    img = tf.image.decode_image(img)
    
    # Resize the image
    img = tf.image.resize(img, size=[img_shape, img_shape])
    
    # Rescale the image to get values to between 0 and 1
    img = img/255.
    
    return img

# Function to predict and display the image with preidcted class
def pred_and_plot(model, img, class_names):
    """
    Predicts the class of the image and displays it.
    :param model: Trained model.
    :param img: Image to predict.
    :param class_names: List of class names.
    returns: None
    """
    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))
    
    # Check if binary or multi-class classification
    if len(class_names) == 2:
        # Get the predicted class
        pred_class = class_names[int(tf.round(pred))]

    else:
        # Get the predicted class
        pred_class = class_names[int(tf.argmax(pred, axis= 1))]
    
    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Prediction: {pred_class}')
    plt.show()

# Function to compare history objects in fine tuning
def compare_history(original_history, new_history, initial_epochs):
  """
  Compares two Tensorflow History objects and plots them.
  :params:
  original_hostory(obj): The history object from initial training
  new_history(obj): The history object from fine tuning
  initial_epochs(int): Number of inital epochs
  Returns:
  None
  """
  # Get original history
  acc = original_history.history["accuracy"]
  loss = original_history.history["loss"]

  val_acc = original_history.history["val_accuracy"]
  val_loss = original_history.history["val_loss"]

  # Combine measurements
  total_acc = acc + new_history.history["accuracy"]
  total_loss = loss + new_history.history["loss"]

  total_val_acc = val_acc + new_history.history["val_accuracy"]
  total_val_loss = val_loss + new_history.history["val_loss"]

  # Make plots
  plt.figure(figsize= (10, 10))
  plt.subplot(2, 1, 1)
  plt.plot(total_acc, label= "Training Accuracy")
  plt.plot(total_val_acc, label= "Validation Accuracy")
  plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label= "Start Fine Tuning")
  plt.legend(loc= "lower right")
  plt.title("Training and Validation Accuracy");

  plt.subplot(2, 1, 2)
  plt.plot(total_loss, label= "Training Loss")
  plt.plot(total_val_loss, label= "Validation Loss")
  plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label= "Start Fine Tuning")
  plt.legend(loc= "lower right")
  plt.title("Training and Validation Loss");

# Function to make predictions for EfficientNet model
def process_predict_plot(model, filename, class_names, img_shape= 224, scale= True):
    """
    Reads image, processes it, makes prediction and plots image.
    :param:
    model(obj): CNN model to use
    filename(str): Name of image file
    class_names(array): Array of class names
    image_shape(int): Integer value of dimension
    scale(bool): Will scale input file from 0-255 to 0-1 if set True.

    Returns:
    None
    """
    # Read in the image
    img = tf.io.read_file(filename)

    # Decode the read file into a tensor
    img = tf.image.decode_image(img)

    # Resize image
    img = tf.image.resize(img, size= [img_shape, img_shape])

    # Check for scaling
    if scale:
        # Scale values
        img = img / 255.

    else:
        pass
  
    # Make prediction
    pred = model.predict(tf.expand_dims(img, axis= 0))

    # Get the predicted class
    pred_class = class_names[int(tf.argmax(pred, axis= 1))]

# Function to preprocess dataset images
def preprocess_images(image, label, img_shape= 224, rescale= False):
    """
    Reads in image and label, converts the dtype to float32 and reshapes the image
    to [image_shape, image_shape, color_channels]
    :param:
        image (tensor): The input image
        label (int): Input label
        image_shape (int): Desired shape of image
        rescale (bool): Flag to rescale image or not

    Returns:
        Tuple of image, label
    """
    # Resize image
    image = tf.image.resize(image, [img_shape, img_shape])

    # Check if rescaling reqiired
    if rescale:
        image = image / 255.
    else:
        pass
    return tf.cast(image, tf.float32), label

# Function to preprocess text data
def preprocess_text_with_line_num(filename):
  """
  Generates list of dictionaries of an abstract's line data.
  :params:
    filename(str): Name of file we need to process
  Returns:
    List of dictionaries.
  """
  # Get all lines from the file
  with open(filename, 'r') as f:
    input_lines =  f.readlines()

  # Empty abstract lines
  abstract_lines = ""

  # Empty list of abstracts
  abstract_samples = []

  # Loop through each line of file
  for line in input_lines:
    # Check to see if line is id line
    if line.startswith("###"):
      abstract_id = line
      # Reset abstract if line is id line
      abstract_lines = ""

    # Check for new line (\n in the abstract will be treated as empty space)
    elif line.isspace():
      # Split abstract into separate lines
      abstract_line_split = abstract_lines.splitlines()

      # Iterate tbrough each line in a single file and count them
      for abstract_line_num, abstract_line in enumerate(abstract_line_split):
        line_data = {}
        target_text_split = abstract_line.split("\t")
        # Get target label
        line_data["target"] = target_text_split[0]
        # Get target text
        line_data["text"] = target_text_split[1].lower()
        # Get line number
        line_data["line_number"] = abstract_line_num
        # Get total lines
        line_data["total_lines"] = len(abstract_line_split) - 1
        # Append to dict
        abstract_samples.append(line_data)

    else:
      # If the line isn't id line, or new line, it contains labelled sentence
      abstract_lines += line

  return abstract_samples

# Function to get and display classification metrics
def get_classification_metrics(y_preds, y_true):
  """
  This function displays the evaluation metrics for classification problem.
  :params:
    y_preds (array): Array containing model predictions
    y_true (array): Array containing ground truth labels

  Returns:
    Dictionary containing accuracy, precision, recall, f1
  """
  # Print out the accuracy score
  acc_score = accuracy_score(y_preds, y_true)
  print(f"The accuracy score is {acc_score:.2f}\n")

  # Print out the precision score
  prec_score = precision_score(y_preds, y_true)
  print(f"The precision score is {prec_score:.2f}\n")

  # Print out the recall score
  rec_score = recall_score(y_preds, y_true)
  print(f"The recall score is {rec_score:.2f}\n")

  # Print out the F1 score
  f_score = f1_score(y_preds, y_true)
  print(f"The F1 score is {f_score:.2f}\n")

  # Plot Confusion Matrix
  ConfusionMatrixDisplay.from_predictions(y_preds, y_true)
  plt.tight_layout();

  return {"accuracy_score": acc_score, "precision_score": prec_score, "recall_score": rec_score, "f1_score": f_score}
