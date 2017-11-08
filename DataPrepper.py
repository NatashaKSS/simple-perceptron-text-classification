# Import necessary modules
from Tokenizer import Tokenizer

#===========================================================================#
# PREPARING THE DATASET FOR TEXT CLASSIFICATION
# Executes the text normalization phase
#===========================================================================#
class DataPrepper():
  def __init__(self, PATH_TO_STOP_WORDS, PATH_TO_TRAIN_CLASS_LIST):
    self.Tokenizer = Tokenizer()
    self.PATH_TO_STOP_WORDS = PATH_TO_STOP_WORDS
    self.PATH_TO_TRAIN_CLASS_LIST = PATH_TO_TRAIN_CLASS_LIST

    # Set up class-specific constants
    self.fpc = self.load_paths_to_training_text() # F.P.C means filename_path_classnames
    self.class_names = self.get_class_names()

    print("[DataPrepper] Instantiated!")

  """
  Train the classifier for the class `pos_class`
  """
  def run(self, class_name):
    print("[DataPrepper] Running...")
    train_data = self.prep_training_set(class_name)

    # text normalization: stop word removal & stemming

    # construct vocabulary from filename_path_classnames

    # sample len(positives) from other len(classes) - 1 classes

    # convert each to feature vector and return them

  def prep_training_set(self, pos_class_name):
    # Get a list of all texts classified as positive first in FPC format
    positives_fpc = self.get_texts_for_class(pos_class_name)

    # Setting our sample sizes
    train_N = int(len(positives_fpc) * 0.80)
    test_N = int(len(positives_fpc) * 0.20)

    # Split the positive classes into train and test sets
    positives = self.sample_N_pos_texts(positives_fpc, train_N)
    train_positives = positives[0]
    test_positives = positives[1]

    # Sample and split the negatives classes into train and test sets
    negative_classes = [class_name for class_name in self.class_names if class_name != pos_class_name]
    negatives = self.sample_N_neg_texts(negative_classes, train_N, test_N)
    train_negatives = negatives[0]
    test_negatives = negatives[1]

    print(len(train_positives.keys()))
    print(len(test_positives.keys()))
    print(len(train_negatives.keys()))
    print(len(test_negatives.keys()))

  """
  Reads the train-class-list or test-class-list file to retrieve all the
  paths to each document

  Returns a list of 3-tuples in the format:
    [[doc_name, path_to_doc, class_name], ...]
  """
  def load_paths_to_training_text(self):
    filepath_class_file = open(self.PATH_TO_TRAIN_CLASS_LIST, 'r')
    filepath_class_lines = filepath_class_file.readlines()

    filename_path_classnames = []
    for ln in filepath_class_lines:
      filepath_class_pair = self.Tokenizer.split_on_whitespace_from_back(ln)
      filename = self.Tokenizer.split_on_slash_from_back(filepath_class_pair[0])[1]
      filepath_class_pair[1] = self.Tokenizer.strip_newline(filepath_class_pair[1])

      result = []
      result.append(filename)
      result.append(filepath_class_pair[0])
      result.append(filepath_class_pair[1])
      filename_path_classnames.append(result)

    return filename_path_classnames

  """
  Gets the list of all the class names in our corpus

  Returns a list of [String] class names
  """
  def get_class_names(self):
    result = []
    for filename_path_classname in self.fpc:
      candidate_class_name = filename_path_classname[2]
      if candidate_class_name not in result:
        result.append(candidate_class_name)
    return result

  """
  Gets a list of filenames classified as `class_name`

  Returns a list of up to LIMIT (optional) 3-tuples in the format:
    [[doc_name, path_to_doc, class_name], ...]
  for the specified class_name
  """
  def get_texts_for_class(self, class_name, LIMIT=None):
    result = []
    for filename_path_classname in self.fpc:
      if filename_path_classname[2] == class_name:
        if LIMIT != None and len(result) > LIMIT:
          break
        else:
          result.append(filename_path_classname)
    return result

  """
  Retrieves the first N texts from a positive class

  Returns a tuple of a
    1.) dictionary of N positive training entries the format:
    2.) dictionary of N positive testing entries the format:

    [
      { '[doc_name]' : 'some long string of text...' ... },
      { '[doc_name]' : 'some long string of text...' ... }
    ]
  """
  def sample_N_pos_texts(self, pos_fpc, N):
    result_train = {}
    result_test = {}
    count = 0

    # Obtain the documents from each class specified in class_names
    # First N documents are sent for training, the remaining are sent for testing
    for fpc in pos_fpc:
      doc_name = fpc[0]
      path_to_doc = fpc[1]
      class_name = fpc[2]

      f = open(path_to_doc, 'r', encoding='latin1')
      if count < N:
        result_train[doc_name] = f.read()
        count += 1
      else:
        result_test[doc_name] = f.read()

    return (result_train, result_test)

  def sample_N_neg_texts(self, negative_classes, N_train, N_test):
    result_train = {}
    result_test = {}
    count = {}

    N_train_per_class = int(N_train / len(negative_classes))
    N_test_per_class = int(N_test / len(negative_classes))

    # Initialize counts for all classes
    for class_name in negative_classes:
      count[class_name] = 0

    # Split N_train into training set and N_test into test set
    for fpc in self.fpc:
      doc_name = fpc[0]
      path_to_doc = fpc[1]
      class_name = fpc[2]

      if class_name in negative_classes and count[class_name] < (N_train_per_class + N_test_per_class):
        f = open(path_to_doc, 'r', encoding='latin1')
        if count[class_name] < N_train_per_class:
          result_train[doc_name] = f.read()
        else:
          result_test[doc_name] = f.read()
        count[class_name] += 1

    print(count)
    return (result_train, result_test)
