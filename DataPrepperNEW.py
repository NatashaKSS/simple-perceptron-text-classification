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
  def __init__(self, PATH_TO_STOP_WORDS, PATH_TO_TRAIN_LIST):
    self.PATH_TO_STOP_WORDS = PATH_TO_STOP_WORDS
    self.PATH_TO_CLASS_LIST = PATH_TO_TRAIN_LIST
    self.Tokenizer = Tokenizer(self.PATH_TO_STOP_WORDS)

    # Set up class-specific constants
    self.fpc = self.load_paths_to_training_text() # F.P.C means filename_path_classnames
    self.class_names = self.get_class_names()

    print("[DataPrepper] Instantiated!")

  """
  Processes the dataset and returns the feature vectors of each of the training
  and test sets (positively and negatively classified)

  Note:
    train_pos_doc_map = datasets[0][0]
    train_neg_doc_map = datasets[0][1]
    test_pos_doc_map = datasets[1][0]
    test_neg_doc_map = datasets[1][1]
  """
  def run(self, class_name, cross_validation_mode=False):
    print("[DataPrepper] Running for", class_name, ", prepping datasets...")

    return []

  #===========================================================================#
  # TEXT NORMALIZATION
  # Functions to facilitate text normalization for all datasets
  #
  # ALSO CONSTRUCTS VOCABULARY & DOC FREQ MAP ON-THE-FLY
  #===========================================================================#
  def tokenize_datasets(self, datasets):
    doc_freq_map = {}

    for i in range(len(datasets)):
      for j in range(len(datasets[i])):
        dict_class_documents = datasets[i][j]

        for doc_name in dict_class_documents.keys():
          dict_class_documents[doc_name] = self.Tokenizer.tokenize(dict_class_documents[doc_name])

          # Construct doc freq map on-the-fly
          tokens_processed_before = []
          for token in dict_class_documents[doc_name]:
            if token not in tokens_processed_before: # unique tokens in a doc
              tokens_processed_before.append(token)
              if token not in doc_freq_map.keys(): # if token is newly found, initialize
                doc_freq_map[token] = [doc_name]
              else:
                doc_freq_map[token].append(doc_name) # since the word appears in this doc

    return [datasets, doc_freq_map]

  #===========================================================================#
  # TF-IDF VECTORIZATION
  # Compute TF-IDF vectors for every document
  #===========================================================================#
  def setup_tfidf_vector(self, NUM_DOCS, datasets, doc_freq_map):
    vocab = list(doc_freq_map.keys())

    for i in range(len(datasets)):
      for j in range(len(datasets[i])):
        dict_class_documents = datasets[i][j]

        for doc_name in dict_class_documents.keys():
          doc = dict_class_documents[doc_name]
          f_vector = [0] * len(vocab)

          for token in doc:
            if token in vocab:
              tf = doc.count(token)
              log_tf = (1 + log(tf)) if tf > 0 else 0.0
              log_idf = log(NUM_DOCS / len(doc_freq_map[token]))
              w = log_tf * log_idf
              f_vector[vocab.index(token)] = w

          dict_class_documents[doc_name] = f_vector

    return datasets

  def cull_doc_freq(self, doc_freq_map, threshold_num_docs):
    culled_df_map = {}
    for word in doc_freq_map.keys():
      if len(doc_freq_map[word]) > threshold_num_docs:
        culled_df_map[word] = doc_freq_map[word]
    return culled_df_map

  #===========================================================================#
  # CONSTRUCT VOCABULARY & DOC FREQ MAP
  # Set up data structures that hold the vocab and doc freq of every word
  #===========================================================================#

  #===========================================================================#
  # CONSTRUCT THE DATASET
  # Retrieves texts from training and test files
  #===========================================================================#

  """
  Reads the train-class-list or test-class-list file to retrieve all the
  paths to each document

  Returns a list of 3-tuples in the format:
    [[doc_name, path_to_doc, class_name], ...]
  """
  def load_paths_to_training_text(self):
    filepath_class_file = open(self.PATH_TO_CLASS_LIST, 'r')
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
  def sample_texts(self, N):
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
