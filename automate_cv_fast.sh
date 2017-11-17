#!/bin/bash
python3 tc-train.py stopword-list train-class-list-blind-2 model2
python3 tc-test.py stopword-list model2 test-list-blind-2 output2
python3 tc-crossvalidation.py stopword-list test-class-list-blind-2 output2

python3 tc-train.py stopword-list train-class-list-blind-4 model4
python3 tc-test.py stopword-list model4 test-list-blind-4 output4
python3 tc-crossvalidation.py stopword-list test-class-list-blind-4 output4
