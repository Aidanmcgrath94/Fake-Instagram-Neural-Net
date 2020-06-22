import pandas as pd
import numpy as np
import time
import sklearn as sk

from sklearn.metrics                import accuracy_score
from sklearn.metrics                import f1_score
from sklearn.metrics                import confusion_matrix

from sklearn.preprocessing          import Normalizer
from sklearn.preprocessing          import StandardScaler  # doctest: +SKIP

from sklearn.naive_bayes            import GaussianNB
from sklearn                        import svm
from sklearn.neural_network         import MLPClassifier

def model_accuracy(start_time, test_y, preds):
    print("Accuracy Score: ",accuracy_score(test_y, preds))
    print("f1 Macro Score: ",f1_score(test_y, preds, average='macro'))
    print("f1 Micro Score: ",f1_score(test_y, preds, average='micro'))
    print("f1 weighted Score: ",f1_score(test_y, preds, average='weighted'))

    print("Confusion Matrix",confusion_matrix(test_y, preds))

    print("Section took %s seconds " % (time.time() - start_time))
    print("  ")
    return

Traindf = pd.read_csv("Datasets\TrainingSet.csv", sep=',', header=0).sample(frac=1).reset_index(drop=True)
Testdf = pd.read_csv("Datasets\TestSet.csv", sep=',', header=0).sample(frac=1).reset_index(drop=True)

train_y = Traindf.iloc[:,-1]
train_X = Traindf.iloc[:, list(range(11))]

test_y = Testdf.iloc[:,-1]
test_X = Testdf.iloc[:, list(range(11))]

#Normalise features - 
train_X = Normalizer().transform(train_X)
test_X = Normalizer().transform(test_X)

#===================================
#       GNB
#===================================
print("---- GNB ----")
start_time = time.time()

# Initialize gnb classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(train_X, train_y)

# Predictions
gnbpreds = gnb.predict(test_X)

# Evaluate accuracy
model_accuracy(start_time, test_y, gnbpreds)

#===================================
#       SVM
#===================================
print("---- SVM ----")
start_time = time.time()

# Initialize svm classifier
SVM = svm.LinearSVC()

# Train our classifier
SVM.fit(train_X, train_y)

# Predictions
svmpreds = SVM.predict(test_X)

# Evaluate accuracy
model_accuracy(start_time, test_y, svmpreds)

#===================================
#       NN
#===================================
print("---- Neural Network ----")
start_time = time.time()

n_samples = len(train_X)

# Initialize Network classifier
NN = MLPClassifier(solver='adam',max_iter = 1500, batch_size=min(20, n_samples))

# Train the classifier
NN.fit(train_X, train_y)

# Predictions
NNpreds = NN.predict(test_X)

# Evaluate accuracy
model_accuracy(start_time, test_y, NNpreds)
