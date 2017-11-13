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

    datasets = None
    if cross_validation_mode:
      datasets = self.prep_dataset(class_name, 1.0, 1.0)
    else:
      datasets = self.prep_dataset(class_name, 0.8, 0.9)

    print("Sample sizes - Train: %d positives + %d negatives, Test: %d positives + %d negatives" %
    (len(datasets[0][0]), len(datasets[0][1]), len(datasets[1][0]), len(datasets[1][1])))

    # Text normalization: tokenization, stop word removal & stemming
    print("[DataPrepper] Tokenizing datasets...")
    datasets_df_pair = self.tokenize_datasets(datasets)
    datasets = datasets_df_pair[0]
    doc_freq_map = datasets_df_pair[1]

    # Construct df from datasets
    doc_freq_map = self.cull_doc_freq(doc_freq_map, 50)
    print("Num of words in vocabs: Vocab=%d" % len(doc_freq_map.keys()))
    print("Num of words in vocabs: Culled Vocab=%d" % len(doc_freq_map.keys()))

    N_docs = len(datasets[0][0]) + len(datasets[0][1]) + len(datasets[1][0]) + len(datasets[1][1])
    datasets = self.setup_tfidf_vector(N_docs, datasets, doc_freq_map)

    # === FOR DEBUGGING ===
    # tryA = datasets[0][0][list(datasets[0][0].keys())[0]]
    # tryB = datasets[0][1][list(datasets[0][1].keys())[0]]
    # tryC = datasets[1][0][list(datasets[1][0].keys())[0]]
    # tryD = datasets[1][1][list(datasets[1][1].keys())[0]]
    # print('---SEE WHAT FEATURE VECTORS LOOK LIKE FOR %s---' % class_name)
    # print('try A:', tryA, 'dim:', len(tryA))
    # print('try B:', tryB, 'dim:', len(tryB))
    # print('try C:', tryC, 'dim:', len(tryC))
    # print('try D:', tryD, 'dim:', len(tryD))
    # print('---END SEE WHAT FEATURE VECTORS LOOK LIKE---')

    f_vector_pos_train = self.setup_feature_vectors_for_classifier(datasets[0][0])
    f_vector_neg_train = self.setup_feature_vectors_for_classifier(datasets[0][1])
    f_vector_pos_test  = []
    f_vector_neg_test  = []
    if cross_validation_mode:
      f_vector_pos_test  = self.setup_feature_vectors_for_classifier(datasets[1][0])
      f_vector_neg_test  = self.setup_feature_vectors_for_classifier(datasets[1][1])

    return [[f_vector_pos_train, f_vector_neg_train], [f_vector_pos_test, f_vector_neg_test], doc_freq_map]

  #===========================================================================#
  # TEXT NORMALIZATION
  # Functions to facilitate text normalization for all datasets
  #===========================================================================#
  def tokenize_datasets_OLD(self, datasets):
    for i in range(len(datasets)):
      for j in range(len(datasets[i])):
        dict_class_documents = datasets[i][j]

        for doc_name in dict_class_documents.keys():
          dict_class_documents[doc_name] = \
            self.Tokenizer.tokenize(dict_class_documents[doc_name])
    return datasets

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
  def setup_vocab(self, dataset, threshold):
    count_vocab = {}
    vocab = []
    for doc_name in dataset.keys():
      for token in dataset[doc_name]:
        if token not in count_vocab.keys():
          count_vocab[token] = 0
        else:
          count_vocab[token] += 1

        if token not in vocab and count_vocab[token] >= threshold:
          vocab.append(token)

    return vocab

  """
  Sets up the doc frequency of words in a given dataset.
  A dataset is a dictionary of this format: { 'doc_name' :  ['Here', 'are', ...] }

  Returns a dictionary containing the document frequency of all words in the
  chosen dataset in this format: { 'Here' : 12, 'are' : 56 ... }
  """
  def setup_doc_freq(self, dataset):
    df = {}

    for doc_name in dataset.keys():
      for word in dataset[doc_name]:
        if word not in df.keys():
          df[word] = [doc_name]
        else:
          if doc_name not in df[word]:
            df[word].append(doc_name)

    return df

  def get_chisq_vocab(self, data_pos_vocab, data_neg_vocab, docs_pos, docs_neg, threshold):
    combined_vocabs = self.union_vocabs(data_pos_vocab, data_neg_vocab)
    N_pos_docs = len(docs_pos.keys())
    N_neg_docs = len(docs_neg.keys())

    feature_selected_vocab = []
    for word in (combined_vocabs):
      N_pos_docs_containing_word = self.get_num_contains_word(docs_pos, word)
      N_pos_docs_not_containing_word = N_pos_docs - N_pos_docs_containing_word

      N_neg_docs_containing_word = self.get_num_contains_word(docs_neg, word)
      N_neg_docs_not_containing_word = N_neg_docs - N_neg_docs_containing_word

      # no. of training docs that:
      N_00 = N_neg_docs_not_containing_word  #  in negative class, do not contain w
      N_01 = N_pos_docs_not_containing_word  #  in positive class, do not contain w
      N_10 = N_neg_docs_containing_word      #  in negative class,        contain w
      N_11 = N_pos_docs_containing_word      #  in positive class,        contain w

      chisq = 0
      if not (N_00 == 0 and N_01 == 0):
        chisq = ((N_11 + N_10 + N_01 + N_00) * pow(N_11 * N_00 - N_10 * N_01, 2)) / \
                ((N_11 + N_01) * (N_11 + N_10) * (N_10 + N_00) * (N_01 + N_00))

      if chisq > threshold:
        feature_selected_vocab.append(word)

    return feature_selected_vocab

  def get_num_contains_word(self, df, word):
    docs_containing_word = []
    for doc_name in df.keys():
      if word in df[doc_name]:
        docs_containing_word.append(doc_name)
    return len(docs_containing_word)

  def union_vocabs(self, vocab_1, vocab_2):
    unioned_vocab = []
    for word in vocab_1:
      if word not in unioned_vocab:
        unioned_vocab.append(word)
    for word in vocab_2:
      if word not in unioned_vocab:
        unioned_vocab.append(word)
    return unioned_vocab

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

      # Normalize by the number of words in a document
      for k in range(len(f_vector)):
        f_vector[k] = f_vector[k] / DOC_N

      # Finished processing a feature vector of a doc
      dataset_f_vectors.append(f_vector)

    return dataset_f_vectors

  """
  Stack map of {'doc_name': [1.81, 0, 6.8...] ... } into a list of feature vectors
  """
  def setup_feature_vectors_for_classifier(self, doc_tfidf_vector_map):
    f_vectors = []
    for doc_name in doc_tfidf_vector_map.keys():
      f_vectors.append(doc_tfidf_vector_map[doc_name])
    return f_vectors

  #===========================================================================#
  # CONSTRUCT THE DATASET
  # Retrieves texts from training and test files
  #===========================================================================#
  """
  Prepares the datasets we will need for training and testing.
  Splits our corpus into positive and negative train/test sets.

  Returns a list of 2 pairs of tuples - one for train & test set, where each
  tuple contains 2 dictionaries - one for positives & negatives
  """
  def prep_dataset(self, positive_class_name, pos_frac, neg_frac_per_class):
    positives_fpc = self.get_texts_for_class(positive_class_name)
    N_pos_docs = len(positives_fpc)

    negatives_fpc_map = {}
    N_neg_docs = 0

    # Set up a dictionary containing { 'neg_class_name': [['53886', 'path_to_doc', 'c2'], [...] ...] }
    for class_name in self.class_names:
      if not (class_name == positive_class_name):
        negatives_fpc_map[class_name] = self.get_texts_for_class(class_name)
        N_neg_docs += 1

    # Split the positive classes into train and test sets
    N_pos_train = int(N_pos_docs * pos_frac)
    N_pos_test = int(N_pos_docs * (1 - pos_frac))

    positives = self.sample_N_pos_texts(positives_fpc, N_pos_train)
    train_positives = positives[0]
    test_positives = positives[1]

    # Sample and split the negatives classes into train and test sets
    negatives = self.sample_N_neg_texts(negatives_fpc_map, neg_frac_per_class)
    train_negatives = negatives[0]
    test_negatives = negatives[1]

    return [[train_positives, train_negatives], [test_positives, test_negatives]]

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
  def sample_N_neg_texts(self, negatives_fpc_map, neg_frac_per_class):
    negative_classes = negatives_fpc_map.keys()
    neg_train_map = {}
    neg_test_map = {}

    for class_name in negative_classes:
      N_docs = len(negatives_fpc_map[class_name])
      N_train = int(N_docs * neg_frac_per_class)

      for i in range(N_docs):
        # Retrieve elements in fpc 3-tuple
        doc_tuple = negatives_fpc_map[class_name][i]
        doc_name = doc_tuple[0]
        path_to_doc = doc_tuple[1]
        class_name = doc_tuple[2]

        f = open(path_to_doc, 'r', encoding='latin1')

        if i < N_train:
          neg_train_map[doc_name] = f.read()
        else:
          neg_test_map[doc_name] = f.read()

    return (neg_train_map, neg_test_map)
