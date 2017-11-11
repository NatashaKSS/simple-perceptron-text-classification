# Import standard modules
import sys
import pickle
from math import log
from math import pow

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

  def run_and_save(self, class_name):
    feature_vectors = self.run(class_name)
    with open('f_vectors_base.pickle', 'wb') as f:
      pickle.dump(feature_vectors, f)

  """
  Processes the dataset and returns the feature vectors of each of the training
  and test sets (positively and negatively classified)
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
    self.print_counts_df(doc_freq_map)
    # self.print_highest_df(doc_freq_map)

    vocab = list(doc_freq_map.keys()) # list of all the words in our corpus
    chisq_vocab = self.get_chisq_vocab(datasets[0], 5)
    print("Num of words in vocabs: Vocab=%d and Chisq_Vocab=%d" % (len(vocab), len(chisq_vocab)))

    # convert each to feature vector and return them
    f_vector_pos_train = self.setup_feature_vectors(chisq_vocab, datasets[0][0])
    f_vector_neg_train = self.setup_feature_vectors(chisq_vocab, datasets[0][1])
    f_vector_pos_test  = self.setup_feature_vectors(chisq_vocab, datasets[1][0])
    f_vector_neg_test  = self.setup_feature_vectors(chisq_vocab, datasets[1][1])

    # print(f_vector_pos_train[:5])

    return [[f_vector_pos_train, f_vector_neg_train], [f_vector_pos_test, f_vector_neg_test]]

  def print_counts_df(self, doc_freq_map):
    count_num_1 = 0
    count_num_5 = 0
    count_num_10 = 0
    for k in doc_freq_map.keys():
      if doc_freq_map[k] <= 1:
        count_num_1 += 1
      elif doc_freq_map[k] > 1 and doc_freq_map[k] <= 5:
        count_num_5 += 1
      else:
        count_num_10 += 1
    print('N<=1:', count_num_1, ' 1<N<=5:', count_num_5, ' N>10:', count_num_10)

  def print_highest_df(self, doc_freq_map):
    for w in sorted(doc_freq_map, key=doc_freq_map.get, reverse=True):
      print(w, doc_freq_map[w])

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
  # CONSTRUCT VOCABULARY & DOC FREQ MAP
  # Set up data structures that hold the vocab and doc freq of every word
  #===========================================================================#
  """
  Sets up the doc frequency of words in a given dataset.
  A dataset is a dictionary of this format: { 'doc_name' :  ['Here', 'are', ...] }

  Returns a dictionary containing the document frequency of all words in the
  chosen dataset in this format: { 'Here' : 12, 'are' : 56 ... }
  """
  def setup_doc_freq(self, dataset):
    THRESHOLD = 2
    df = {}

    for doc_name in dataset.keys():
      for word in dataset[doc_name]:
        if word not in df.keys():
          df[word] = [doc_name]
        else:
          if doc_name not in df[word]:
            df[word].append(doc_name)

    result = {}
    for word in df.keys():
      doc_freq = len(df[word])
      if (doc_freq > THRESHOLD):
        result[word] = doc_freq

    return result

  def get_chisq_vocab(self, datasets_train, threshold):
    data_pos = datasets_train[0]
    N_pos = len(data_pos.keys())
    data_neg = datasets_train[1]
    N_neg = len(data_neg.keys())

    data_pos_doc_freq = self.setup_doc_freq(data_pos)
    data_pos_vocab = list(data_pos_doc_freq.keys())
    data_neg_doc_freq = self.setup_doc_freq(data_neg)
    data_neg_vocab = list(data_neg_doc_freq.keys())

    feature_selected_vocab = []
    for word in (data_pos_vocab + data_neg_vocab):
      # no. of training docs that:
      N_00 = 0 #  in negative class, do not contain w
      N_01 = 0 #  in positive class, do not contain w
      N_10 = 0 #  in negative class,        contain w
      N_11 = 0 #  in positive class,        contain w

      pos_word = data_pos_doc_freq[word] if word in data_pos_vocab else 0
      neg_word = data_neg_doc_freq[word] if word in data_neg_vocab else 0

      N_00 = float(N_neg - neg_word)
      N_01 = float(N_pos - pos_word)
      N_10 = float(neg_word)
      N_11 = float(pos_word)

      if not (N_10 == N_neg or N_11 == N_pos):
        chisq = ((N_11 + N_10 + N_01 + N_00) * pow(N_11 * N_00 - N_10 * N_01, 2)) / \
                ((N_11 + N_01) * (N_11 + N_10) * (N_10 + N_00) * (N_01 + N_00))
        if chisq > threshold:
          feature_selected_vocab.append(word)
      else:
        chisq = 0

    return feature_selected_vocab

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

      for k in range(len(f_vector)):
        value = f_vector[k] / DOC_N
        f_vector[k] = value

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
