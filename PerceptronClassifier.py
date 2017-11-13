#===========================================================================#
# THE PERCEPTRON LEARNER
# Executes the training & classification phase
#===========================================================================#
class PerceptronClassifier():
  def __init__(self):
    print("[PerceptronClassifier] Instantiated!")
    self.learning_rate = 0.1
    self.num_epochs = 50

  """
  Trains a weight vector using the perceptron learning algorithm
  """
  def train(self, X, y_true):
    w = [1.0] + [0.0] * (len(X[0]) - 1) # first term is always the bias
    for epoch in range(self.num_epochs):
      n_errors = 0 # accumulate number of errors in this epoch
      for x, y in zip(X, y_true):
        update = self.learning_rate * (y - self.classify(x, w, self.threshold_activation))
        n_errors += int(update != 0.0)

        # Update bias and weights
        w[0] += update
        for i in range(len(x) - 1):
          w[i + 1] += update * x[i]

      print('epoch=%d, error=%.d' % (epoch + 1, n_errors))
    return w

  """
  Determines if a specified feature vector is of class 1 or 0 given a learned
  weight vector.

  Pass in an activation function (use the ones given in the section
  'Activation Functions' below)
  """
  def classify(self, x, w, activation_func, debug_mode=False):
    activation = w[0]
    for i in range(len(x) - 1):
      activation += w[i + 1] * x[i]
    if debug_mode:
      print('activation:', activation)
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
