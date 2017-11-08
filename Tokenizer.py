# Import necessary modules
from porter import PorterStemmer

#===========================================================================#
# PREPARING THE DATASET FOR TEXT CLASSIFICATION
# Executes the text normalization phase
#===========================================================================#
class Tokenizer():
  def __init__(self, PATH_TO_STOP_WORDS):
    print("[Tokenizer] Instantiated!")
    self.PATH_TO_STOP_WORDS = PATH_TO_STOP_WORDS
    self.STOP_WORDS = self.load_stopwords()
    self.PorterStemmer = PorterStemmer()

  def tokenize(self, sentence):
    print("[Tokenizer] Running...")

  def stem(self, word):
    return self.PorterStemmer.stem(word, 0, len(word) - 1)

  def remove_stopwords(self, tokens):
    return list(filter(lambda tok: tok not in self.STOP_WORDS, tokens))

  #===========================================================================#
  # STRING MANIPULATION FUNCS
  #===========================================================================#

  """
  Split on 1st whitespace from back
  """
  def split_on_whitespace_from_back(self, input_str):
    return input_str.rsplit(' ', 1)

  """
  Split on 1st '/' from back
  """
  def split_on_slash_from_back(self, input_str):
    return input_str.rsplit('/', 1)

  """
  Trim newline char from an input string
  """
  def strip_newline(self, input_str):
    return input_str.strip('\n')

  #===========================================================================#
  # SETUP
  #===========================================================================#
  def load_stopwords(self):
    f = open(self.PATH_TO_STOP_WORDS, 'r')
    stopwords = f.read().splitlines()
    return stopwords
