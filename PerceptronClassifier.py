#===========================================================================#
# THE PERCEPTRON LEARNER
# Executes the training & classification phase
#===========================================================================#
class PerceptronClassifier():
  def __init__(self):
    print("[PerceptronClassifier] Instantiated!")

  def train(self, train_vectors):
    print("[PerceptronClassifier] Training...")
    x = self.setup_feature_vectors(train_vectors)
    w = self.learn_weights(x, 0.01, 5)
    print(w)
    print(len(w), len(x[0]))
    print(self.classify(x[0], w, self.threshold_activation))
    print(self.classify(x[456], w, self.threshold_activation))

  """
  Trains a weight vector using the perceptron learning algorithm
  """
  def learn_weights(self, x, learning_rate, num_epochs):
    w = [0] * len(x[0])
    for epoch in range(num_epochs):
      for feature_vector in x:
        y = self.classify(feature_vector, w, self.threshold_activation)
        error = feature_vector[-1] - y

        # Update bias and weights
        w[0] = w[0] + learning_rate * error
        for i in range(len(feature_vector) - 1):
          w[i + 1] = w[i + 1] + learning_rate * feature_vector[i] * error
      print('epoch=%d' % epoch)
    return w

  """
  Determines if a specified feature vector is of class 1 or 0 given a learned
  weight vector.

  Pass in an activation function (use the ones given in the section
  'Activation Functions' below)
  """
  def classify(self, f_vector, w, activation_func):
    activation = w[0]
    for i in range(len(f_vector) - 1):
      activation += w[i + 1] * f_vector[i]
    return activation_func(activation)

  #===========================================================================#
  # Activation Functions
  # All of them return 1 for the positive class and 0 for the negative class
  #===========================================================================#
  """
  Simple threshold activation function
  """
  def threshold_activation(self, activation):
    return 1 if activation >= 0.0 else 0

  #===========================================================================#
  # A bit of pre-processing here to get feature vectors into the correct shape
  #===========================================================================#
  """
  Sets feature_vectors into the correct shape, with its true y classification
  appended to the back of the feature vector's list
  """
  def setup_feature_vectors(self, train_vectors):
    pos_feature_vectors = train_vectors[0]
    neg_feature_vectors = train_vectors[1]

    for f_vector_pos in pos_feature_vectors:
      f_vector_pos.append(1.0)

    for f_vector_neg in neg_feature_vectors:
      f_vector_neg.append(0.0)

    return pos_feature_vectors + neg_feature_vectors
