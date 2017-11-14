# Import standard modules
import sys
import pickle

# Import necessary modules
from Tokenizer import Tokenizer

#===================================================================#
# TESTING THE TEXT CLASSIFIER
# Executes the testing phase of the text classifiers on documents
# given in test-list.
#
# Run with command:
#   python3 tc-crossvalidation.py stopword-list true-class-list path-to-predicted
#
# true-class-list is where we have the filepaths and their true classes
# path-to-predicted is where we have the filepaths and their predicted classes
#===================================================================#
class TCCrossValidation():
  def __init__(self):
    print("[Tester] instantiated!")
    self.Tokenizer = Tokenizer(PATH_TO_STOP_WORDS)

  def validate_accuracy(self):
    print("[TCCrossValidation] Computing accuracy...")
    true_filename_path_classnames = self.load_paths_to_text(PATH_TO_TRAIN_CLASS_LIST)
    predicted_filename_path_classnames = self.load_paths_to_text(PATH_TO_TEST_CLASS_LIST)
    print('Accuracy:', self.compute_accuracy(true_filename_path_classnames, predicted_filename_path_classnames))

  def compute_accuracy(self, y_true, y_predict):
    correct = 0
    N = len(y_true)
    for i in range(N):
      if y_true[i][2] == y_predict[i][2]:
        correct += 1
    return correct / N

  """
  Reads test-class-list file

  Returns a list of 3-tuples in the format:
    [[doc_name, path_to_doc, class_name], ...]
  """
  def load_paths_to_text(self, PATH_TO_CLASS_LIST):
    filepath_class_file = open(PATH_TO_CLASS_LIST, 'r')
    filepath_class_lines = filepath_class_file.readlines()

    filename_path_classnames = []
    for ln in filepath_class_lines:
      filepath_class_pair = self.Tokenizer.split_on_whitespace_from_back(ln)
      filepath = filepath_class_pair[0]
      filename = self.Tokenizer.split_on_slash_from_back(filepath)[1]
      class_name = self.Tokenizer.strip_newline(filepath_class_pair[1])

      filename_path_classnames.append([filename, filepath, class_name])

    return filename_path_classnames

#===========================================================================#
# EXECUTING THE PROGRAM
#===========================================================================#
PATH_TO_STOP_WORDS = sys.argv[1]
PATH_TO_TRAIN_CLASS_LIST = sys.argv[2]
PATH_TO_TEST_CLASS_LIST = sys.argv[3]

print("PATH_TO_STOP_WORDS:", PATH_TO_STOP_WORDS,
      ", PATH_TO_TRAIN_CLASS_LIST", PATH_TO_TRAIN_CLASS_LIST,
      ", PATH_TO_TEST_CLASS_LIST", PATH_TO_TEST_CLASS_LIST)

TCCrossValidation().validate_accuracy()
