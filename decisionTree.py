from sklearn import tree
import numpy as np
from MnistExperiment.getData import load

a=load()
for i in range(len(a)):
        print(a[i].shape)
print('----'*10)
x_train =a[0]
y_train =a[1]
x_test =a[2]
y_test =a[3]

classifier = tree.DecisionTreeClassifier()
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

classifier.fit(x_train,y_train)
score = classifier.score(x_test,y_test)
print(score)

