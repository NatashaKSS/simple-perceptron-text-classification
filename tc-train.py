# Import standard modules
import sys
import pickle
from DataPrepper import DataPrepper
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
    feature_vectors_classes_docfreq = self.DataPrepper.run()
    feature_vectors_classes = feature_vectors_classes_docfreq[0]
    self.insert_bias(feature_vectors_classes)
    print('Dim of feature vector:', len(feature_vectors_classes[0][0]))
    doc_freq_map = feature_vectors_classes_docfreq[1]

    # For all classes in class_names, train a perceptron
    for class_name in class_names:
      f_train_vectors = self.setup_feature_vectors(class_name, feature_vectors_classes)
      X = f_train_vectors[0]
      y = f_train_vectors[1]
      w = self.PerceptronClassifier.train(X, y, learning_rate=0.02, num_epochs=50)
      print('weight:', w)

      weight_docfreq_map[class_name] = w
      print('=== FINISHED TRAINING MODEL FOR CLASS %s ===\n\n' % class_name)

    self.save_models([weight_docfreq_map, doc_freq_map])

  def validate(self):
    print("[TextClassifier] Validating perceptron classifier...")

    # Get all class names to make a perceptron classifier for
    class_names = self.DataPrepper.class_names

    # Setup feature vectors for corpus
    feature_vectors_classes_docfreq = self.DataPrepper.run()
    feature_vectors_classes = feature_vectors_classes_docfreq[0]
    doc_freq_map = feature_vectors_classes[1]

    # For all classes in class_names, train a perceptron
    for class_name in class_names:
      f_vectors_mixed = self.setup_feature_vectors(class_name, feature_vectors_classes)
      f_train_test_vectors = self.setup_feature_vectors_split(f_vectors_mixed[0], f_vectors_mixed[1], 200)
      f_train_vectors = f_train_test_vectors[0]
      f_test_vectors = f_train_test_vectors[1]
      X_train = f_train_vectors[0]
      y_train = f_train_vectors[1]
      X_test = f_test_vectors[0]
      y_test = f_test_vectors[1]

      print("Sample sizes - Train: %d samples, Test: %d samples" % (len(X_train), len(X_test)))
      w = self.PerceptronClassifier.train(X_train, y_train, learning_rate=1.0, num_epochs=100)
      acc = self.PerceptronClassifier.batch_classify_with_acc(w, X_test, y_test, debug_mode=False)
      print('weight:', w)
      print('Accuracy:', acc)
      print('=== FINISHED VALIDATING MODEL FOR CLASS %s ===\n\n\n' % class_name)

  def save_models(self, models_df):
    print("[TextClassifier] Saving model to disk...")
    pickle.dump(models_df, open(PATH_TO_MODEL, 'wb'))

  def insert_bias(self, f_vectors_classnames):
    for f_vector_classname in f_vectors_classnames:
      f_vector_classname[0].insert(0, 1.0)

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

      # Separate the feature vector from its labelled class_name
      result_f_vectors.append(f_vector)

      # Re-mapping classnames to positive or negative classes
      if y_true == pos_class_name:
        y.append(1) # because positive
      else:
        y.append(-1) # because all other classes other than pos_class_name are negative

    return [result_f_vectors, y]

  def setup_feature_vectors_split(self, f_vectors, y, num_test):
    num_pos = int(num_test / 2)
    num_neg = num_test - num_pos

    result_f_train_vectors = []
    y_train = []

    result_f_test_vectors = []
    y_test = []

    count_num_pos = 0
    count_num_neg = 0
    for i, f_vector in enumerate(f_vectors):
      if (count_num_pos < num_pos) and y[i] == 1:
        result_f_test_vectors.append(f_vector)
        y_test.append(y[i])
        count_num_pos += 1
      elif (count_num_neg < num_neg) and y[i] == -1:
        result_f_test_vectors.append(f_vector)
        y_test.append(y[i])
        count_num_neg += 1
      else:
        # Remaining goes into training set
        result_f_train_vectors.append(f_vector)
        y_train.append(y[i])

    return [[result_f_train_vectors, y_train], [result_f_test_vectors, y_test]]

#===========================================================================#
# EXECUTING THE PROGRAM
#===========================================================================#
PATH_TO_STOP_WORDS = sys.argv[1]
PATH_TO_TRAIN_CLASS_LIST = sys.argv[2]
PATH_TO_MODEL = sys.argv[3]

print("PATH_TO_STOP_WORDS:", PATH_TO_STOP_WORDS,
      ", PATH_TO_TRAIN_CLASS_LIST:", PATH_TO_TRAIN_CLASS_LIST,
      ", PATH_TO_MODEL", PATH_TO_MODEL)

TextClassifier().build()

# pickle.dump(model, open(PATH_TO_MODEL, 'wb'))
print("=== FINISHED TRAINING...MODEL SAVED IN " + PATH_TO_MODEL + " ===")
