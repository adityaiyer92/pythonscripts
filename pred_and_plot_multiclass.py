def pred_and_plot_multiclass(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction with a multiclass 
  model and plots an image with predicted class with title.
  """

  # Imports the target image and preprocesses it
  img = load_and_prep_image(filename)

  # Make prediction
  pred = model.predict(tf.expand_dims(img, axis= 0))

  # Get the predicted class
  pred_class = class_names[int(tf.argmax(pred, axis= 1))]

  # Plot the image with Predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False);
