# Import standard modules
import os
import sys
import numpy as np

# Import necessary modules
from Tokenizer import Tokenizer

#===================================================================#
# GENERATES THE FILES FOR CROSS VALIDATION
#===================================================================#

def load_paths_from_master():
  filepath_class_file = open('train-class-list', 'r')
  filepath_class_lines = filepath_class_file.readlines()

  filename_class_pairs = {}

  for ln in filepath_class_lines:
    filepath_class_pair = tokenizer.split_on_whitespace_from_back(ln)
    filepath = filepath_class_pair[0]
    class_name = filepath_class_pair[1].strip('\n')

    if filename_class_pairs.get(class_name):
      filename_class_pairs[class_name].append([filepath, class_name])
    else:
      filename_class_pairs[class_name] = [[filepath, class_name]]

  return filename_class_pairs

def split_into(filename_class_pairs):
  for batch in range(5):
    batch_train_and_test = batch_train_and_test_sets(filename_class_pairs, batch)
    train = batch_train_and_test[0]
    test = batch_train_and_test[1]

    print('Writing to batch', str(batch))
    with open('train-class-list-blind-' + str(batch), 'w') as f:
      for pair in train:
        filepath = pair[0]
        class_name = pair[1]

        f.write(filepath + ' ' + class_name + '\n')

    with open('test-class-list-blind-' + str(batch), 'w') as f:
      for pair in test:
        filepath = pair[0]
        class_name = pair[1]

        f.write(filepath + ' ' + class_name + '\n')

    with open('test-list-blind-' + str(batch), 'w') as f:
      for pair in test:
        filepath = pair[0]

        f.write(filepath + '\n')

def batch_train_and_test_sets(filename_class_pairs, batch_index):
  train = []
  test = []

  for class_name in filename_class_pairs:
    shifted_filename_class_pairs = np.roll(filename_class_pairs[class_name], batch_index * 100)
    class_train = np.array(shifted_filename_class_pairs[:400]).tolist()
    class_test = np.array(shifted_filename_class_pairs[-100:]).tolist()

    train = train + class_train
    test = test + class_test

  return [train, test]

tokenizer = Tokenizer('stopword-list')
filename_class_pairs = load_paths_from_master()
split_into(filename_class_pairs)
