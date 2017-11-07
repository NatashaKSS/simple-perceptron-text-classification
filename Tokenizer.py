# Import necessary modules
from porter import PorterStemmer

#===========================================================================#
# PREPARING THE DATASET FOR TEXT CLASSIFICATION
# Executes the text normalization phase
#===========================================================================#
class Tokenizer():
  def __init__(self):
    print("[Tokenizer] Instantiated!")
    self.PorterStemmer = PorterStemmer()

  def tokenize(self, sentence):
    print("[Tokenizer] Running...")

  def stem(self, word):
    return self.PorterStemmer.stem(word, 0, len(word) - 1)
