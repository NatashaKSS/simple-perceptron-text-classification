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

    # For all classes in class_names, train a perceptron
    for class_name in class_names:
      if class_name == 'c1': # TODO: HARDCODED FOR TESTING - REMEMBER TO ITERATE THROUGH EVERY CLASS
        # feature_vectors = self.DataPrepper.run(class_name)
        # train_vectors = feature_vectors[0] # [f_vector_pos_train, f_vector_neg_train]
        # test_vectors = feature_vectors[1]  # [f_vector_pos_test, f_vector_neg_test]

        # self.train(train_vectors)
        self.train([[ [1,2,3], [2,4,6] ], [ [6,7,8], [8,9,10] ]])

  def train(self, train_vectors):
    print("[TextClassifier] Training perceptron classifier...")
    self.PerceptronClassifier.train(train_vectors)

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
