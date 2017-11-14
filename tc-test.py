# Import standard modules
import sys
import pickle
import numpy as np
from DataPrepper import DataPrepper
from PerceptronClassifier import PerceptronClassifier

#===================================================================#
# TESTING THE TEXT CLASSIFIER
# Executes the testing phase of the text classifiers on documents
# given in test-list.
#
# Run with command:
#   python3 tc-test.py stopword-list model test-list test-class-list
#
# test-class-list is where we store the results of our classification
# test-list is where we take in the paths to the documents we want to classify
#===================================================================#
class TCTest():
  def __init__(self):
    print("[Tester] instantiated!")
    self.DataPrepper = DataPrepper(PATH_TO_STOP_WORDS, PATH_TO_TEST_LIST, test_mode=True)
    models_df = self.load_models_df()
    self.models = models_df[0]
    self.df = models_df[1]

  def test(self):
    print("[Tester] Prepping dataset...")
    print("[TextClassifier] Prepping dataset...")

    # Setup feature vectors for corpus
    feature_vectors_filepath = self.DataPrepper.run_test(self.df)
    f_vectors = [x[0] for x in feature_vectors_filepath]
    f_vectors = self.add_bias_to_f_vectors(f_vectors)
    filepaths = [x[1] for x in feature_vectors_filepath]

    # CLASSIFICATION
    y_predict = self.classify(f_vectors)

    self.output(filepaths, y_predict)

  def add_bias_to_f_vectors(self, f_vectors):
    for f_vector in f_vectors:
      # Insert bias term
      f_vector.insert(0, 1.0)
    return f_vectors

  def classify(self, f_vectors):
    y_predict = []
    for f_vector in f_vectors:
      y_predict.append(self.get_best_model(f_vector))
    return y_predict

  def get_best_model(self, f_vector):
    class_names = list(self.models.keys())

    score_so_far = 0
    best_class_so_far = class_names[0]
    scores = []
    for class_name in self.models.keys():
      score = np.dot(self.models[class_name], f_vector)
      scores.append([class_name, score])
      if score > score_so_far:
        score_so_far = score
        best_class_so_far = class_name
    print(list(reversed(sorted(scores, key=lambda x: x[1]))))
    return best_class_so_far

  def load_models_df(self):
    return pickle.load(open(PATH_TO_MODEL, 'rb'))

  def output(self, filepaths, y_predict):
    print("[TextClassifier] Saving output to", PATH_TO_TEST_CLASS_LIST)
    with open(PATH_TO_TEST_CLASS_LIST, 'w') as f:
      for i in range(len(y_predict)):
        f.write(filepaths[i] + ' ' + y_predict[i] + '\n')

  """
  Reads the test-class-list file to retrieve all the paths to each document

  Returns a list of y_true in the format:
    { 'doc_name': 'c1' ... }
  """
  def load_paths_to_testing_text(self):
    filepath_class_file = open(PATH_TO_TEST_CLASS_LIST, 'r')
    filepath_class_lines = filepath_class_file.readlines()

    filename_path_classnames = {}
    for ln in filepath_class_lines:
      filepath_class_pair = self.Tokenizer.split_on_whitespace_from_back(ln)
      filename = self.Tokenizer.split_on_slash_from_back(filepath_class_pair[0])[1]
      filepath_class_pair[1] = self.Tokenizer.strip_newline(filepath_class_pair[1])

      filename_path_classnames[filename] = filepath_class_pair[1]

    return filename_path_classnames


#===========================================================================#
# EXECUTING THE PROGRAM
#===========================================================================#
PATH_TO_STOP_WORDS = sys.argv[1]
PATH_TO_MODEL = sys.argv[2]
PATH_TO_TEST_LIST = sys.argv[3]
PATH_TO_TEST_CLASS_LIST = sys.argv[4]

print("PATH_TO_STOP_WORDS:", PATH_TO_STOP_WORDS,
      ", PATH_TO_MODEL:", PATH_TO_MODEL,
      ", PATH_TO_TEST_LIST", PATH_TO_TEST_LIST,
      ", PATH_TO_TEST_CLASS_LIST", PATH_TO_TEST_CLASS_LIST)

TCTest().test()

print("=== FINISHED TESTING...RESULTS SAVED IN " + PATH_TO_TEST_CLASS_LIST + " ===")
