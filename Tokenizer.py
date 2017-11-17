# Import necessary modules
import re
from porter import PorterStemmer

PUNCTUATIONS = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'

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

  """
  Tokenizes on these rules:
    SPLIT INTO TOKENS
    STRIP TOKENS' WHITESPACES, NEWLINES AND PUNCTUATIONS DANGLING IN BETWEEN
    STEM EVERY TOKEN
    REMOVE TOKEN IF IS STOP WORD

  Returns list of text normalized tokens
  """
  def tokenize(self, input_str):
    result = []
    # input_str_list = input_str.split()
    input_str_list = re.split('\W+', input_str)

    for token in input_str_list:
      result_tok = token.strip(PUNCTUATIONS)
      if len(result_tok) > 1 and \
         not self.is_stopword(result_tok.lower()) and \
         not self.isMixedNumeric(result_tok):

        result_tok = self.stem(result_tok)
        result.append(result_tok.lower())

    return result

  def stem(self, word):
    return self.PorterStemmer.stem(word, 0, len(word) - 1)

  def remove_stopwords(self, tokens):
    return list(filter(lambda tok: tok not in self.STOP_WORDS, tokens))

  def is_stopword(self, token):
    return self.STOP_WORDS.get(token)

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

  """
  Determines whether an input string has the RegEx given in this function
  A RegEx match object will be returned if a complete match occurs
  """
  def isMixedNumeric(self, input_str):
    pattern = r'([0-9]+[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]*)+'
    return re.match(pattern, input_str)

  #===========================================================================#
  # SETUP
  #===========================================================================#
  def load_stopwords(self):
    f = open(self.PATH_TO_STOP_WORDS, 'r')
    stopwords = f.read().splitlines()

    stopword_dict = {}
    for word in stopwords:
      stopword_dict[word] = True

    return stopword_dict
