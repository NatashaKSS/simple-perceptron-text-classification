# Import necessary modules
from Tokenizer import Tokenizer

#===========================================================================#
# PREPARING THE DATASET FOR TEXT CLASSIFICATION
# Executes the text normalization phase
#===========================================================================#
class DataPrepper():
  def __init__(self):
    print("[DataPrepper] Instantiated!")
    self.Tokenizer = Tokenizer()

  def run(self):
    print("[DataPrepper] Running...")
