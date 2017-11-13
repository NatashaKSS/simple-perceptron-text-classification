# Import standard modules
import sys
import pickle
from DataPrepper import DataPrepper
from PerceptronClassifier import PerceptronClassifier

#===========================================================================#
# TESTING THE TEXT CLASSIFIER
# Executes the testing phase of the text classifiers on documents given in
# test-list.
#
# Run with command:
#   python3 tc-train.py stopword-list train-class-list model
#===========================================================================#
class TCTest():
  def __init__(self):
    print("[Tester] instantiated!")
    self.DataPrepper = DataPrepper(PATH_TO_STOP_WORDS, PATH_TO_TEST_LIST)

  def test(self):
    print("[Tester] Prepping dataset...")

  def output(self):
    print("[TextClassifier] Saving output to", PATH_TO_TEST_CLASS_LIST)

#===========================================================================#
# EXECUTING THE PROGRAM
#===========================================================================#
PATH_TO_STOP_WORDS = sys.argv[1]
PATH_TO_MODEL = sys.argv[2]
PATH_TO_TEST_LIST = sys.argv[3]
PATH_TO_TEST_CLASS_LIST = sys.argv[4]

print("PATH_TO_STOP_WORDS:", PATH_TO_STOP_WORDS,
      ", PATH_TO_MODEL:", PATH_TO_MODEL,
      ", PATH_TO_TEST_LIST", PATH_TO_TEST_LIST)
      ", PATH_TO_TEST_CLASS_LIST", PATH_TO_TEST_CLASS_LIST)

TCTest().test()

print("=== FINISHED TESTING...RESULTS SAVED IN " + PATH_TO_TEST_CLASS_LIST + " ===")
