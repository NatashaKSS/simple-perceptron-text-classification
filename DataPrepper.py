# Import standard modules
import sys
from math import log

# Import necessary modules
from Tokenizer import Tokenizer

#===========================================================================#
# PREPARING THE DATASET FOR TEXT CLASSIFICATION
# Executes the text normalization phase
#===========================================================================#
class DataPrepper():
  def __init__(self, PATH_TO_STOP_WORDS, PATH_TO_TRAIN_CLASS_LIST):
    self.PATH_TO_STOP_WORDS = PATH_TO_STOP_WORDS
    self.PATH_TO_TRAIN_CLASS_LIST = PATH_TO_TRAIN_CLASS_LIST
    self.Tokenizer = Tokenizer(self.PATH_TO_STOP_WORDS)

    # Set up class-specific constants
    self.fpc = self.load_paths_to_training_text() # F.P.C means filename_path_classnames
    self.class_names = self.get_class_names()

    print("[DataPrepper] Instantiated!")

  """
  Train the classifier for the class `pos_class`
  """
  def run(self, class_name):
    print("[DataPrepper] Running...")
    datasets = self.prep_dataset(class_name)

    # text normalization: tokenization, stop word removal & stemming
    datasets = self.tokenize_datasets(datasets)
    # print(datasets[0][0]['37261']) # c1 train
    # print(datasets[0][1]['60140']) # c2 train
    # print(datasets[1][0]['38677']) # c1 test
    # print(datasets[1][1]['60174']) # c2 test

    # construct vocabulary from datasets
    doc_freq_map = self.setup_doc_freq(dict(datasets[0][0], **datasets[0][1]))
    vocab = list(doc_freq_map.keys()) # list of all the words in our corpus

    # convert each to feature vector and return them
    f_vector_pos_train = self.setup_feature_vectors(vocab, datasets[0][0])
    f_vector_neg_train = self.setup_feature_vectors(vocab, datasets[0][1])
    f_vector_pos_test = self.setup_feature_vectors(vocab, datasets[1][0])
    f_vector_neg_test = self.setup_feature_vectors(vocab, datasets[1][1])
    print(len(f_vector_pos_train))
    print(len(f_vector_neg_train))
    print(len(f_vector_pos_test))
    print(len(f_vector_neg_test))

  #===========================================================================#
  # TEXT NORMALIZATION
  # Functions to facilitate text normalization for all datasets
  #===========================================================================#
  def tokenize_datasets(self, datasets):
    for i in range(len(datasets)):
      for j in range(len(datasets[i])):
        dict_class_documents = datasets[i][j]
        for doc_name in dict_class_documents.keys():
          dict_class_documents[doc_name] = \
            self.Tokenizer.tokenize(dict_class_documents[doc_name])
    return datasets

  #===========================================================================#
  # CONSTRUCT VOCABULARY
  # Set up data structures that hold the doc freq of every word in our corpus
  #===========================================================================#
  def setup_doc_freq(self, dataset):
    df = {}

    for doc_name in dataset.keys():
      for word in dataset[doc_name]:
        if word not in df.keys():
          df[word] = [doc_name]
        else:
          if doc_name not in df[word]:
            df[word].append(doc_name)

    for word in df.keys():
      df[word] = len(df[word])

    return df

  #===========================================================================#
  # CONSTRUCT FEATURE VECTORS FOR EACH CLASS
  # Compute feature vectors representing each class' text document
  #===========================================================================#
  def setup_feature_vectors(self, vocab, dataset):
    fea_datasets = []
    dataset_f_vectors = []

    for doc_name in dataset.keys():
      doc = dataset[doc_name]
      DOC_N = len(doc)
      f_vector = [0] * len(vocab)

      # Count word occurrence with reference to vocab
      for word in doc:
        if word in vocab:
          f_vector[vocab.index(word)] += 1

      # log normalize term frequencies
      # all values will be negative but the Highest score still implies that a term occurred the most often.
      # Lowest score is about -708.3964185322641, if that term never occurred.
      for k in range(len(f_vector)):
        value = float(f_vector[k]) / float(DOC_N)
        f_vector[k] = log(value) if value != 0.0 else log(sys.float_info.min)

      # Finished processing a feature vector of a doc
      dataset_f_vectors.append(f_vector)

    return dataset_f_vectors

  #===========================================================================#
  # CONSTRUCT THE DATASET
  # Retrieves texts from training and test files
  #===========================================================================#

  """
  Prepares the datasets we will need for training and testing

  Splits our corpus into positive and negative train/test sets.

  Returns a list of 2 pairs of tuples - one for train & test set, where each
  tuple contains 2 dictionaries - one for positives & negatives
  """
  def prep_dataset(self, pos_class_name):
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

    return [(train_positives, train_negatives), (test_positives, test_negatives)]

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
    1.) dictionary of N positive training entries,
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

  """
  Retrieves the first N / len(negative_classes) texts from each of the
  specified list of negative classes

  Returns a tuple of a
    1.) dictionary of N negative training entries,
    2.) dictionary of N negative testing entries the format:

    [
      { '[doc_name]' : 'some long string of text...' ... },
      { '[doc_name]' : 'some long string of text...' ... }
    ]
  """
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

    return (result_train, result_test)
