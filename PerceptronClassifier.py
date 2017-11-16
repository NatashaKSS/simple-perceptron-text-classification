import math
import numpy as np

#===========================================================================#
# THE PERCEPTRON LEARNER
# Executes the training & classification phase
#===========================================================================#
class PerceptronClassifier():
  def __init__(self):
    print("[PerceptronClassifier] Instantiated!")

  """
  Trains a weight vector using the perceptron learning algorithm
  """
  def train(self, X, y_true, learning_rate=0.1, num_epochs=50):
    N_dim = len(X[0])
    N_samples = len(X)
    w = np.ones(N_dim)

    for epoch in range(num_epochs):
      n_errors = 0 # accumulate number of errors in this epoch

      for i in range(N_samples):
        x = X[i]
        y = y_true[i]
        update = learning_rate * (y - self.classify(x, w, self.sigmoid_threshold_activation))
        n_errors += int(update != 0.0)

        # Update weights
        w += np.multiply(update, x)

      if n_errors <= 1: # 1 is enough to stop
        break;

      print('epoch=%d, error=%.d' % (epoch + 1, n_errors))
    return w

  """
  Determines if a specified feature vector is of class 1 or -1 given a learned weight vector.
  Pass in an activation function (use the ones given in the section 'Activation Functions' below)
  """
  def classify(self, x, w, activation_func, debug_mode=False):
    activation = np.dot(x, w)
    return activation_func(activation)

  #===========================================================================#
  # Activation Functions
  # All of them return 1 for the positive class and 0 for the negative class
  #===========================================================================#
  """
  Simple threshold activation function
  """
  def threshold_activation(self, activation):
    return 1 if activation >= 0.0 else -1

  """
  Sigmoid activation function
  """
  def sigmoid_threshold_activation(self, gamma):
    if gamma < 0:
      sig = 1 - 1 / (1 + math.exp(gamma))
    else:
      sig = 1 / (1 + math.exp(-gamma))
    return 1 if sig >= 0.5 else -1

  """
  Tanh activation function
  """
  def tanh_threshold_activation(self, gamma):
    sig = np.tanh(gamma)
    return 1 if sig >= 0.5 else -1

  #===========================================================================#
  # Classification Functions
  #===========================================================================#
  def batch_classify_with_acc(self, w, X, y_true, debug_mode=False):
    y_predict = self.batch_classify(w, X, debug_mode)
    print('y_true:', len(y_true), 'y_predict:', len(y_predict))
    print('--- y_true ---')
    print(y_true)
    print('--- y_predict ---')
    print(y_predict)

    return self.compute_acc(y_true, y_predict)

  def batch_classify(self, w, X, debug_mode=False):
    y_predict = []
    for x in X:
      if debug_mode:
        print('x:', x)
        print('weight:', w)

      y_predict.append(self.classify(x, w, self.threshold_activation, debug_mode=False))

      if debug_mode:
        print('---')

    return y_predict

  def compute_acc(self, y_true, y_predict):
    correct = 0
    N = len(y_true)
    for i in range(N):
      if y_true[i] == y_predict[i]:
        correct += 1
    return correct / N
