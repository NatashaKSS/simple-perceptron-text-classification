# CS4248 Assignment 3 - Perceptron Text Classification
Performs text classification using the Perceptron Learning Algorithm.
This application is general enough to work on *any number of classes*, *any class names* 
and *any number of training texts* within a class.

### Data Preparation / Text Normalization
* Stop word removal
* Stemming using Porter's Stemmer
* Case-folding -(tentative)-

### Feature selection for dimensionaltiy reduction
*Rules:*
* Select a stemmed word as a feature for a class `c` if it has high chi-squared value
* Select a stemmed word with a high enough inverse document frequency -(tentative)-

### Perceptron Learning Step
Note: This is a multi-class perceptron learner where 1 classifier is learned for each 
class. Each text is assumed to belong to exactly one of the given classes.

### Instructions
#### Train the text classifier:
```
python tc-train.py stopword-list train-class-list model
```
where `model` is the file where we will stored our learned perceptron weights. `stopword-list` is a file 
containing a list of stop words. `train-class-list` is a file containing the following lines:
```
/home/course/cs4248/tc/c1/37261 c1
/home/course/cs4248/tc/c1/37913 c1
/home/course/cs4248/tc/c1/37914 c1
...
/home/course/cs4248/tc/c1/58343 c1
```

#### Run text classifier on given assignment test set:
```
python tc-test.py stopword-list model test-list test-class-list
```
where `stopword-list` is the same file containing a list of stop words. `model` is the file of 
weights learned during training. `test-list` is a file that contains a list of the locations 
of test texts to be classified as such:
```
/home/course/cs4248/tc/test/001
/home/course/cs4248/tc/test/002
/home/course/cs4248/tc/test/003
...
```
and `test-class-list` is a file in the same format as `train-class-list`.


