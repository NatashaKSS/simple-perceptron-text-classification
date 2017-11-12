# Import standard modules
import sys
from DataPrepper import DataPrepper
from PerceptronClassifierNEW import PerceptronClassifierNEW

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
    self.PerceptronClassifier = PerceptronClassifierNEW()

  def build(self):
    print("[TextClassifier] Prepping dataset...")

    # Get all class names to make a perceptron classifier for
    class_names = self.DataPrepper.class_names

    # For all classes in class_names, train a perceptron
    for class_name in class_names:
      if class_name == 'c1': # TODO: HARDCODED FOR TESTING - REMEMBER TO ITERATE THROUGH EVERY CLASS
        # When you want to train from some pickled feature vectors
        # feature_vectors = []
        # with open('f_vectors_base.pickle', 'rb') as handle:
        #   feature_vectors = pickle.load(handle)

        # When you want to pre-process the dataset again
        feature_vectors = self.DataPrepper.run(class_name)
        train_vectors = feature_vectors[0] # [f_vector_pos_train, f_vector_neg_train]
        test_vectors = feature_vectors[1]  # [f_vector_pos_test, f_vector_neg_test]

        # When you wanna save the feature vectors
        # self.DataPrepper.run_and_save(class_name) # When you want to save the model

        # self.train(train_vectors)
        self.validateNEW(train_vectors, test_vectors)

  def train(self, train_vectors):
    print("[TextClassifier] Training perceptron classifier...")
    return self.PerceptronClassifier.train(train_vectors)

  def validate(self, train_vectors, test_vectors):
    print("[TextClassifier] Validating perceptron classifier...")
    w = self.PerceptronClassifier.train(train_vectors)
    print('weight:', w)
    acc = self.PerceptronClassifier.batch_classify_with_acc(w, test_vectors)
    print(acc)

  def validateNEW(self, train_vectors, test_vectors):
    print("[TextClassifier] Validating perceptron classifier...")
    f_train_vectors = self.setup_feature_vectors(train_vectors)
    X = f_train_vectors[0]
    y = f_train_vectors[1]

    f = self.setup_feature_vectors([[[2.7810836, 2.550537003], [1.465489372, 2.362125076], [3.396561688, 4.400293529], [1.38807019, 1.850220317], [3.06407232, 3.005305973]], [[7.627531214, 2.759262235], [5.332441248, 2.088626775], [6.922596716, 1.77106367], [8.675418651, -0.242068655], [7.673756466, 3.508563011]]])
    X = f[0]
    y = f[1]
    print(X)
    print(y)
    w = self.PerceptronClassifier.train(X, y)
    print('weight:', w)

    f_test_vectors = self.setup_feature_vectors(test_vectors)
    X_test = f_test_vectors[0]
    y_test = f_test_vectors[1]
    acc = self.PerceptronClassifier.batch_classify_with_acc(w, X, y)
    print('Accuracy:', acc)

  def saveModel(self):
    print("[TextClassifier] Saving model to disk...")

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
  def setup_feature_vectors(self, f_vectors):
    pos_feature_vectors = f_vectors[0]
    neg_feature_vectors = f_vectors[1]

    f_vectors = []
    y = []

    for f_vector_pos in pos_feature_vectors:
      f_vector = [1.0] # with bias term
      for elem in f_vector_pos:
        f_vector.append(elem)
      f_vectors.append(f_vector)
      y.append(1) # because positive

    for f_vector_neg in neg_feature_vectors:
      f_vector = [1.0] # with bias term
      for elem in f_vector_neg:
        f_vector.append(elem)
      f_vectors.append(f_vector)
      y.append(-1) # because negative

    return [f_vectors, y]

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
model.build()
model.saveModel()

# pickle.dump(model, open(PATH_TO_MODEL, 'wb'))
print("=== FINISHED TRAINING...MODEL SAVED IN " + PATH_TO_MODEL + " ===")
