import numpy as np
from os import listdir
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import k_means

path = "./feature/train/"
model_path = "./model/"
test_path = "./feature/test/"

test_accuracy = []
N = 31


# 读txt文件并将每个文件的描述子改为一维的矩阵存储
def txtToVector(filename):
    returnVec = np.zeros((1, N))
    fr = open(filename)
    lineStr = fr.readline()
    lineStr = lineStr.split(' ')
    for i in range(N):
        returnVec[0, i] = int(lineStr[i])
    return returnVec


def get_train_data():
    labels = []  # 存放类别标签
    trainingFileList = listdir(path)
    m = len(trainingFileList)
    data = np.zeros((m, N))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        labels.append(classNumber)
        data[i, :] = txtToVector(path + fileNameStr)  # 将训练集改为矩阵格式
    return data, labels


def train_model():
    data, labels = get_train_data()
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    model = RandomForestClassifier()

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    score = accuracy_score(y_predict, y_test)

    print('{}% of samples were classified correctly !'.format(score * 100))

    return model


def save_model(model, model_name):
    joblib.dump(model, f'./model/{model_name}.pkl', compress=9)


## training model & test
if __name__ == "__main__":
    data, labels = get_train_data()
    model = train_model()
    # save_model(model=model, model_name='RandomForest')
