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
  def __init__(self, PATH_TO_STOP_WORDS, PATH_TO_CLASS_LIST, test_mode=False):
    self.PATH_TO_STOP_WORDS = PATH_TO_STOP_WORDS
    self.PATH_TO_CLASS_LIST = PATH_TO_CLASS_LIST
    self.Tokenizer = Tokenizer(self.PATH_TO_STOP_WORDS)

    # Set up class-specific constants
    if test_mode:
      # F.P.C means filename_path_classnames, there aren't any classes retrieved
      # during the test phase, but we keep the name convention for convenience
      self.fpc = self.load_paths_to_test_text()
    else:
      # F.P.C means filename_path_classnames
      self.fpc = self.load_paths_to_training_text()
      self.class_counts = self.get_class_counts()
      self.class_names = list(self.class_counts.keys())

    print("[DataPrepper] Instantiated!")

  """
  Processes the dataset and returns their feature vectors in the format:
    [[feature_vector, 'c1'], [feature_vector, 'c2'] ...]
  """
  def run(self):
    print("[DataPrepper] Running on training set...")

    print("[DataPrepper] Sampling texts from disk...")
    dataset = self.sample_texts()

    print("[DataPrepper] Tokenizing...")
    doc_df_pair = self.tokenize_dataset(dataset)
    docs = doc_df_pair[0]
    doc_freq = doc_df_pair[1]
    doc_freq = self.cull_doc_freq(doc_freq, 5, len(doc_freq.keys()))
    doc_freq = self.cull_low_chisq_doc_freq(doc_freq, 4)
    print("Number of words in vocab:", len(doc_freq.keys()))
    print("[DataPrepper] Setting up feature vectors...")
    feature_vectors_class = self.setup_tfidf_vectors(docs, doc_freq)

    # Debug
    print(feature_vectors_class[0])
    print(feature_vectors_class[501])
    print(feature_vectors_class[1101])
    print(feature_vectors_class[1601])
    print(feature_vectors_class[2051])

    return [feature_vectors_class, doc_freq]

  def run_test(self, doc_freq):
    print("[DataPrepper] Running on testset...")
    dataset_filepath = self.sample_texts_for_test()
    doc_df_pair = self.tokenize_dataset_for_test(dataset_filepath)
    docs = doc_df_pair[0]
    doc_freq_testset = doc_df_pair[1]
    f_vectors_filepath = self.setup_tfidf_vectors(docs, doc_freq, doc_freq_map_testset=doc_freq_testset, test_mode=True)
    print(f_vectors_filepath[0])
    print(f_vectors_filepath[20])
    print(f_vectors_filepath[45])
    return f_vectors_filepath

  #===========================================================================#
  # TEXT NORMALIZATION
  # Functions to facilitate text normalization for all datasets
  #
  # ALSO CONSTRUCTS VOCABULARY / DOC FREQ MAP ON-THE-FLY
  #===========================================================================#
  def tokenize_dataset(self, dict_class_documents):
    doc_freq_map = {}
    class_df_map = {}
    docs = dict_class_documents.keys()
    N_DOCS = len(docs)

    self.print_loading_bar(0, N_DOCS, progress_text='Tokenizing: ', complete_text='Complete')
    for i, doc_name in enumerate(docs):
      dict_class_documents[doc_name][0] = self.Tokenizer.tokenize(dict_class_documents[doc_name][0])
      class_name = dict_class_documents[doc_name][1]

      # Construct doc freq map on-the-fly
      tokens_processed_before = {}
      for token in dict_class_documents[doc_name][0]:
        if not tokens_processed_before.get(token): # unique tokens in a doc
          tokens_processed_before[token] = True # processed before, so mark as True
          if not doc_freq_map.get(token): # if token is newly found, initialize
            # Initialize token's doc freqs
            doc_freq_map[token] = {
              'count': 1,
              'class-specific': {}
            }

            # Initialize class-specific doc freq
            for c in self.class_names:
              doc_freq_map[token]['class-specific'][c] = 0
            doc_freq_map[token]['class-specific'][class_name] = 1

          else:
            doc_freq_map[token]['count'] += 1 # since the word appears in this doc
            doc_freq_map[token]['class-specific'][class_name] += 1

      self.print_loading_bar(i + 1, N_DOCS, progress_text='Tokenizing: ', complete_text='Complete')

    return [dict_class_documents, doc_freq_map]

  def tokenize_dataset_for_test(self, dict_class_documents):
    doc_freq_map = {}
    docs = dict_class_documents.keys()
    N_DOCS = len(docs)

    self.print_loading_bar(0, N_DOCS, progress_text='Tokenizing: ', complete_text='Complete')
    for i, doc_name in enumerate(docs):
      dict_class_documents[doc_name][0] = self.Tokenizer.tokenize(dict_class_documents[doc_name][0])

      # Construct doc freq map on-the-fly
      tokens_processed_before = {}
      for token in dict_class_documents[doc_name][0]:
        if not tokens_processed_before.get(token): # unique tokens in a doc
          tokens_processed_before[token] = True # processed before, so mark as True
          if not doc_freq_map.get(token):   # if token is newly found, initialize
            doc_freq_map[token] = 1
          else:
            doc_freq_map[token] += 1 # since the word appears in this doc

      self.print_loading_bar(i + 1, N_DOCS, progress_text='Tokenizing: ', complete_text='Complete')

    return [dict_class_documents, doc_freq_map]


  #===========================================================================#
  # TF-IDF VECTORIZATION
  # Compute TF-IDF vectors for every document
  #===========================================================================#
  """
  Returns [[f_vector1, class_name], [f_vector2, class_name] ...].
  class_name is path to text doc represented by feature vector if test_mode=True
  """
  def setup_tfidf_vectors(self, dict_class_documents, doc_freq_map, doc_freq_map_testset=None, test_mode=False):
    vocab = list(doc_freq_map.keys())
    doc_names = dict_class_documents.keys()
    N_VOCAB = len(vocab)
    N_DOCNAMES = len(doc_names)
    f_vectors_classname = []

    self.print_loading_bar(0, N_DOCNAMES, progress_text='Setting up feature vectors:', complete_text='Complete')
    for i, doc_name in enumerate(doc_names):
      doc = dict_class_documents[doc_name][0]
      class_name = dict_class_documents[doc_name][1]
      f_vector = [0] * N_VOCAB

      for token in doc:
        if doc_freq_map.get(token):
          tf = doc.count(token)
          log_tf = (1 + log(tf)) if tf > 0 else 0.0

          if test_mode and doc_freq_map_testset:
            log_idf = log(N_DOCNAMES / doc_freq_map_testset[token])
          else:
            log_idf = log(N_DOCNAMES / doc_freq_map[token]['count'])

          w = log_tf * log_idf
          f_vector[vocab.index(token)] = w

      f_vectors_classname.append([f_vector, class_name])

      self.print_loading_bar(i + 1, N_DOCNAMES, progress_text='Setting up feature vectors:', complete_text='Complete')

    return f_vectors_classname

  def cull_doc_freq(self, doc_freq_map, low_num_docs, high_num_docs):
    culled_df_map = {}
    for word in doc_freq_map:
      num_occurrences = doc_freq_map[word]['count']
      if num_occurrences < high_num_docs and num_occurrences > low_num_docs:
        culled_df_map[word] = doc_freq_map[word]
    return culled_df_map

  def cull_low_chisq_doc_freq(self, doc_freq_map, min_chisq):
    print('Culling low chisq...')
    N_total = self.class_counts['total']
    N_classes = len(self.class_names)

    culled_df_map = {}

    for token in doc_freq_map:
      chisq_score = 0

      for pos_class_name in self.class_names:
        # Num of docs in positive class and not in positive class
        N_docs_pos_c = self.class_counts[pos_class_name]
        N_docs_not_in_pos_c = N_total - N_docs_pos_c

        # N00 is the number of training texts that do not contain w and are not in class c.
        # N01 is the number of training texts that do not contain w and are in class c.
        # N10 is the number of training texts that contain w and are not in class c.
        # N11 is the number of training texts that contain w and are in class c.
        N_11 = doc_freq_map[token]['class-specific'][pos_class_name]
        N_01 = N_docs_pos_c - N_11
        N_10 = self.sum_neg_doc_freq(pos_class_name, doc_freq_map[token]['class-specific'])
        N_00 = N_docs_not_in_pos_c - N_10

        if  (N_11 > 0 and N_01 > 0) and \
            (N_11 > 0 and N_10 > 0) and \
            (N_10 > 0 and N_00 > 0) and \
            (N_01 > 0 and N_00 > 0):
          chisq_score += \
            ((N_11 + N_10 + N_01 + N_00) * pow(N_11 * N_00 - N_10 * N_01, 2)) / \
            ((N_11 + N_01) * (N_11 + N_10) * (N_10 + N_00) * (N_01 + N_00))
        else:
          chisq_score += 0

        #print(token, N_docs_pos_c, N_docs_not_in_pos_c, pos_class_name, N_00, N_01, N_10, N_11, doc_freq_map[token]['class-specific'])
      #print(token, chisq_score / N_classes, doc_freq_map[token]['class-specific'])

      average_chisq = chisq_score / N_classes

      # Only accept words into our vocab if their chisquared score is above threshold
      if average_chisq > min_chisq:
        culled_df_map[token] = doc_freq_map[token]

    return culled_df_map

  def sum_neg_doc_freq(self, pos_class_name, token_df):
    sum = 0
    for c in token_df:
      if not c == pos_class_name:
        sum += token_df[c]
    return sum

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
  Reads the test-list file to retrieve all the paths to each test document

  Returns a list of 3-tuples in the format:
    [[doc_name, path_to_doc], ...]
  """
  def load_paths_to_test_text(self):
    filename_path = open(self.PATH_TO_CLASS_LIST, 'r')
    filename_path_lines = filename_path.readlines()

    filename_paths = []
    for ln in filename_path_lines:
      filename = self.Tokenizer.split_on_slash_from_back(ln)[1]
      filename = self.Tokenizer.strip_newline(filename)
      filepath = self.Tokenizer.strip_newline(ln)
      filename_paths.append([filename, filepath])

    return filename_paths

  """
  Gets counts of all docs in each class in our corpus

  Returns a dictionary of class counts
  """
  def get_class_counts(self):
    result = {}
    for filename_path_classname in self.fpc:
      candidate_class_name = filename_path_classname[2]

      if candidate_class_name not in result:
        result[candidate_class_name] = 1
      else:
        result[candidate_class_name] += 1
    result['total'] = len(self.fpc)
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
  Retrieves dictionary of training entries from self.fpc in the format:
    {
      'doc_name1' : ['some long string of this text...', class_name],
      'doc_name2' : ['some long string of this text...', class_name],
      ...
    }
  """
  def sample_texts(self):
    result = {}

    for fpc in self.fpc:
      doc_name = fpc[0]
      path_to_doc = fpc[1]
      class_name = fpc[2]

      f = open(path_to_doc, 'r', encoding='latin1')
      result[doc_name] = [f.read(), class_name]

    return result

  """
  Retrieves dictionary of test entries from self.fpc in the format:
    {
      'doc_name1' : ['some long string of this text...'],
      'doc_name2' : ['some long string of this text...'],
      ...
    }
  """
  def sample_texts_for_test(self):
    result = {}

    for fpc in self.fpc:
      doc_name = fpc[0]
      path_to_doc = fpc[1]

      f = open(path_to_doc, 'r', encoding='latin1')
      result[doc_name] = [f.read(), path_to_doc]

    return result


  """
  Prints a progress bar
  """
  def print_loading_bar(self, chunk, N, progress_text = '', complete_text = ''):
    percentage = (chunk / N) * 100
    percentage_int = int(percentage)
    percentage_decimal = str(percentage - percentage_int)[2]
    bar = '█' * percentage_int + '-' * (100 - percentage_int)
    print('\r%s |%s| %s.%s%% %s' % (progress_text, bar, percentage_int, percentage_decimal, complete_text), end = '\r')

    if percentage >= 100.0:
      print()
