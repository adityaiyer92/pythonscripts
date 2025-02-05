'''
Aditya Iyer
Plot the training and testing loss curves and accuracy curves separately.
'''
def plot_loss_curves(history):
  """
  Returns separate Loss Curves for the Training and Validation/Test datasets
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