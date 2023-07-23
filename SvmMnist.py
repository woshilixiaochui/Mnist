import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn import svm
from MnistExperiment.getData import load

#引入数据集
a=load()
x_train_image = a[0]
y_train_label = a[1]
x_test_image = a[2]
y_test_label = a[3]
#转化为一维向量，长度为784,并设置为float数
x_Train = x_train_image.reshape(60000,784).astype('float')
x_Test = x_test_image.reshape(10000,784).astype('float')

#将数据归一化
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255

#传递训练参数
clf = svm.SVC(C=100,kernel='rbf',gamma=0.03)

#进行模型训练
clf.fit(x_Train_normalize,y_train_label)
predictions = [int(a) for a in clf.predict(x_Test_normalize)]
#混淆矩阵
print(confusion_matrix(y_test_label,predictions))
print(classification_report(y_test_label,np.array(predictions)))

#计算精准度
print('accuracy=',accuracy_score(y_test_label,predictions))


