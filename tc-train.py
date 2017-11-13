# Import standard modules
import sys
import pickle
from DataPrepperNEW import DataPrepper
from PerceptronClassifier import PerceptronClassifier

#===========================================================================#
# TRAINING THE TEXT CLASSIFIER
# Executes the training phase of the text classifier on documents given in
# train-class-list. Saves the trained perceptron weights into the file
# called 'model'.
#
# Run with command:
#   python3 tc-train.py stopword-list train-class-list model
#===========================================================================#
class TextClassifier():
  def __init__(self):
    print("[TextClassifier] instantiated!")
    self.DataPrepper = DataPrepper(PATH_TO_STOP_WORDS, PATH_TO_TRAIN_CLASS_LIST)
    self.PerceptronClassifier = PerceptronClassifier()

  def build(self):
    print("[TextClassifier] Prepping dataset...")

    # Get all class names to make a perceptron classifier for
    class_names = self.DataPrepper.class_names

    # Initialize data structure to save doc freq and weights
    weight_docfreq_map = {}

    # Setup feature vectors for corpus
    feature_vectors_classes = self.DataPrepper.run()

    # For all classes in class_names, train a perceptron
    for class_name in class_names:
      if class_name == 'c1':
        f_train_vectors = self.setup_feature_vectors(class_name, feature_vectors_classes)
        X = f_train_vectors[0]
        y = f_train_vectors[1]
        w = self.PerceptronClassifier.train(X, y, learning_rate=0.5, num_epochs=200)
        print('weight:', w)

        weight_docfreq_map[class_name] = [w, doc_freq_map]
        print('=== FINISHED TRAINING MODEL FOR CLASS %s ===\n\n\n' % class_name)

  def validate(self):
    print("[TextClassifier] Validating perceptron classifier...")

    # Get all class names to make a perceptron classifier for
    class_names = self.DataPrepper.class_names

    # Setup feature vectors for corpus
    feature_vectors_classes = self.DataPrepper.run()

    # For all classes in class_names, train a perceptron
    for class_name in class_names:
      if class_name == 'c1':
        f_train_vectors = self.setup_feature_vectors(class_name, feature_vectors_classes)
        X = f_train_vectors[0]
        y = f_train_vectors[1]
        w = self.PerceptronClassifier.train(X, y, learning_rate=0.01, num_epochs=70)
        acc = self.PerceptronClassifier.batch_classify_with_acc(w, X[:50] + X[-50:], y[:50] + y[-50:], debug_mode=False)
        print('weight:', w)
        print('Accuracy:', acc)
        print('=== FINISHED VALIDATING MODEL FOR CLASS %s ===\n\n\n' % class_name)

  def save_models(self, weight_docfreq_map):
    print("[TextClassifier] Saving model to disk...")
    pickle.dump(weight_docfreq_map, open(PATH_TO_MODEL, 'wb'))

  """
  Sets feature_vectors into the correct shape, with its true y classification
  appended to the back of the feature vector's list

  Returns a tuple of X and y,
    where X is of the format with length n_samples:
    [[feature_vector_of_doc_1], [feature_vector_of_doc_2]...]

    where y is of the format with length n_samples, representing each feature
    vector's true classification:
    [1, 1, 0, ...]
  """
  def setup_feature_vectors(self, pos_class_name, f_vectors_classnames):
    result_f_vectors = []
    y = []

    for f_vector_classname in f_vectors_classnames:
      f_vector = f_vector_classname[0]
      y_true = f_vector_classname[1]

      # Insert bias term
      f_vector.insert(0, 1.0)
      result_f_vectors.append(f_vector)

      # Re-mapping classnames to positive or negative classes
      if y_true == pos_class_name:
        y.append(1) # because positive
      else:
        y.append(-1) # because all other classes other than pos_class_name are negative

    return [result_f_vectors, y]

#===========================================================================#
# EXECUTING THE PROGRAM
#===========================================================================#
PATH_TO_STOP_WORDS = sys.argv[1]
PATH_TO_TRAIN_CLASS_LIST = sys.argv[2]
PATH_TO_MODEL = sys.argv[3]

print("PATH_TO_STOP_WORDS:", PATH_TO_STOP_WORDS,
      ", PATH_TO_TRAIN_CLASS_LIST:", PATH_TO_TRAIN_CLASS_LIST,
      ", PATH_TO_MODEL", PATH_TO_MODEL)

model = TextClassifier()
model.validate()

# pickle.dump(model, open(PATH_TO_MODEL, 'wb'))
print("=== FINISHED TRAINING...MODEL SAVED IN " + PATH_TO_MODEL + " ===")
