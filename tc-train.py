# Import standard modules
import sys
import pickle
from DataPrepper import DataPrepper

#===========================================================================#
# TRAINING THE TEXT CLASSIFIER
# Executes the training phase of the text classifier on documents given in
# train-class-list. Saves the trained perceptron weights into the file
# called 'model'.
#
# Run with command:
#   python tc-train.py stopword-list train-class-list model
#===========================================================================#
class TextClassifier():
  def __init__(self):
    print("[TextClassifier] instantiated!")
    self.DataPrepper = DataPrepper()

  def prepare(self):
    print("[TextClassifier] Prepping dataset...")

  def train(self):
    print("[TextClassifier] Training perceptron classifier...")

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
model.prepare()
model.train()
model.saveModel()

# pickle.dump(model, open(PATH_TO_MODEL, 'wb'))
print("=== FINISHED TRAINING...MODEL SAVED IN " + PATH_TO_MODEL + " ===")
