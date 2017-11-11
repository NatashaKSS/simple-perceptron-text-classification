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
    w = self.learn_weights(x, 0.1, 150)
    return w

  def batch_classify_with_acc(self, w, test_vectors):
    y_true = self.get_true_y(test_vectors)
    y_predict = self.batch_classify(w, test_vectors)
    print('y_true:', len(y_true), 'y_predict:', len(y_predict))
    return self.compute_acc(y_true, y_predict)

  def batch_classify(self, w, test_vectors):
    x = self.setup_feature_vectors(test_vectors)
    y_predict = []

    for feature_vector in x:
      y_predict.append(self.classify(feature_vector, w, self.threshold_activation))

    return y_predict

  def get_true_y(self, f_vectors):
    pos_feature_vectors = f_vectors[0]
    neg_feature_vectors = f_vectors[1]

    y_true = []
    for f_vector in pos_feature_vectors:
      y_true.append(1)

    for f_vector in neg_feature_vectors:
      y_true.append(0)

    return y_true

  def compute_acc(self, y_true, y_predict):
    correct = 0.0
    N = len(y_true)
    for i in range(N):
      if y_true[i] == y_predict[i]:
        correct += 1.0
    return correct / N

  """
  Trains a weight vector using the perceptron learning algorithm
  """
  def learn_weights(self, x, learning_rate, num_epochs):
    w = [1.0] * len(x[0])
    for epoch in range(num_epochs):
      squared_error = 0.0
      for feature_vector in x:
        y_predict = self.classify(feature_vector, w, self.threshold_activation)
        error = feature_vector[-1] - y_predict
        squared_error += error * error

        # Update bias and weights
        w[0] = w[0] + learning_rate * error
        for i in range(len(feature_vector) - 1):
          w[i + 1] = w[i + 1] + learning_rate * feature_vector[i] * error

      print('epoch=%d, error=%.3f' % (epoch + 1, squared_error))
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
    return 1.0 if activation >= 0.0 else 0.0

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

    f_vectors = []

    for f_vector_pos in pos_feature_vectors:
      f_vector = []
      for elem in f_vector_pos:
        f_vector.append(elem)
      f_vector.append(1.0)
      f_vectors.append(f_vector)

    for f_vector_neg in neg_feature_vectors:
      f_vector = []
      for elem in f_vector_neg:
        f_vector.append(elem)
      f_vector.append(0.0)
      f_vectors.append(f_vector)

    return f_vectors
