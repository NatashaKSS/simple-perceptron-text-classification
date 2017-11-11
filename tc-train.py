# Import standard modules
import sys
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
        self.validate(train_vectors, test_vectors)

  def train(self, train_vectors):
    print("[TextClassifier] Training perceptron classifier...")
    return self.PerceptronClassifier.train(train_vectors)

  def validate(self, train_vectors, test_vectors):
    print("[TextClassifier] Validating perceptron classifier...")
    w = self.PerceptronClassifier.train(train_vectors)
    print('weight:', w)
    acc = self.PerceptronClassifier.batch_classify_with_acc(w, test_vectors)
    print(acc)

  def saveModel(self):
    print("[TextClassifier] Saving model to disk...")

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
