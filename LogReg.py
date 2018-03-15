import numpy
from sklearn.model_selection import KFold
import sklearn as sk
#print(sk.__name__)
import matplotlib.pyplot as plt

#REFERENCES: https://beckernick.github.io/logistic-regression-from-scratch/
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.show.html

data = numpy.genfromtxt('MNIST_CVHW3.csv', delimiter=',', dtype=int, skip_header=1)
#print(data.shape[0], 1) Returns (8200, 1)


kf = KFold(n_splits = 10)
#print(kf)

#Split the data with KFold
for train, test in kf.split(data):
    # trainingData, testData = data[train], data[test]   #data
    Xytest = numpy.array(data[test])   #data
    # trainingLabels, testLabels = data[train], data[test]    #Labels
    Xytraining = numpy.array(data[train])    #Labels
    #print(testData)
    #print(test)
    #intercept = numpy.ones((trainingData.shape[0], 1))
    #print(intercept)
    #features = numpy.hstack((intercept, trainingData))
    #print(features)

testingLabels = numpy.array(Xytest[:, 0])
testingData = numpy.array(Xytest[:, 1: -1])
trainingLabels = numpy.array(Xytraining[:, 0])
trainingData = numpy.array(Xytraining[:, 1: -1])

# print(testingData[0])
# print(testingLabels[0])

# Normalize the data knowing the values are from 0 to 255
def normalizeData(dataNorm):
    return dataNorm / 255

#If label = 6 then 0, if label = 8 then 1
#Class 1: 6 and Class 2: 8
def makeBinary(data):  #data = labels
    zeroArray = numpy.zeros(len(data))
    for label in range(len(data)):
        if data[label] == 8:
            zeroArray[label] = 1
    return zeroArray

# Normalize Dataset and Labels
normTestData = normalizeData(testingData)
normTrainData = normalizeData(trainingData)
normTestLabels = makeBinary(testingLabels)
normTrainingLabels = makeBinary(trainingLabels)

# Cost function
def cost(X, y, b):
    return numpy.sum((numpy.dot(X, b) - numpy.array(y))**2)

# likelihood function
def likelihood(data, labels, bvalue):
     scores = numpy.dot(data, bvalue)    #Takes the dot product of the data and bestimate
     summation = numpy.sum(labels * scores - numpy.log(1 + numpy.exp(scores)))
     return summation

# This function makes the prediction using sigmond
def prediction (X, b):
    return sigmoid(numpy.dot(X, b))
# print(prediction(trainingData, b_opt(trainingData, trainingLabels)))

# Classify the test data with threshold of 0.5 (Accuracy)
def accuracy(bestFitLine, testLabels):
    bestFitLine[bestFitLine > 0.5] = 1
    bestFitLine[bestFitLine <= 0.5] = 0
    value = sum(bestFitLine == testLabels) / float(len(testLabels))
    return value

#Accuracy = (TP+TN)/(TP+FP+FN+TN)
def TPTNaccuracy(TP, TN, FP, FN):
    return (TP + TN)/(TP + FP + FN + TN)

def CalcAccuracy(predictionLabels, normTestLabels):
    predictionLabels = (predictionLabels > .5)
    trueCases = sum(normTestLabels)
    falseCases = len(normTestLabels) - trueCases
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for x in range(len(predictionLabels)):
        if predictionLabels[x] == normTestLabels[x]:
            if predictionLabels[x] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if (predictionLabels[x] == 1 and normTestLabels[x] == 0):
                FP += 1
            else:
                FN += 1
    print(trueCases)
    print(falseCases)
    TPR = TP / float(trueCases)
    FPR = FP / float(falseCases)
    return TPR, FPR

# testArray = [1,1,1,1,0]
# labelArray = [0,1,0,1,0]
# r, n = CalcAccuracy(testArray, labelArray)
# print("{}, {}".format(r, n))

#Gradient Descent
def gradientDescent(X, y, b):
    #reference https://github.com/michelucci/Logistic-Regression-Explained/blob/master/MNIST%20with%20Logistic%20Regression%20from%20scratch.ipynb
    predictions = prediction(X, b)
    m = X.shape[0]
    error_cost = (predictions - y).transpose()
    #return -1 / 7800 * numpy.dot(X.transpose(), y - prediction(X, b))
    return -1.0 / m * numpy.dot(X.transpose(), error_cost)

#Counts the number of true data
def trueCount(data):
    return sum(data)

#Counts the number of false data
def falseCount(data):
    return (len(data) - trueCount(data))

#sigmoid function
def sigmoid(value):
    return 1/(1.0 + numpy.exp(-value))

#Get a array of ones
def onesMatrix(data):
    return numpy.ones(data.shape[0], 1)

#Get a array of zero
def zeroMatrix(data):
    return numpy.zeros(data.shape[1])

def TruePositive(predictionLabels, normTestLabels):
    return numpy.sum(numpy.logical_and(predictionLabels, normTestLabels))

def TrueNegative(predictionLabels, normTestLabels):
    return len(normTestLabels) - numpy.sum(numpy.logical_or(predictionLabels, normTestLabels))

def FalsePositive(predictionLabels, normTestLabels):
    return numpy.sum(numpy.logical_and(predictionLabels == 1, normTestLabels == 0))

def FalseNegative(predictionLabels, normTestLabels):
    return numpy.sum(numpy.logical_and(predictionLabels == 0, normTestLabels == 1))

def binaryPrediciton(predictionData):
    #array = predictionData
    for label in range(len(predictionData)):
        if predictionData[label] >= 0.5:
            predictionData[label] = 1
        else:
            predictionData[label] = 0
    return predictionData

# Getting the b value (weights)
bvalue = numpy.zeros(len(normTrainData[0]))  # weights
learningRate = 1e-5
iterator = 1000

for x in range(0, iterator):
    bvalue = bvalue + learningRate * gradientDescent(normTrainData, normTrainingLabels, bvalue)

# # Prediction with Gradient Descent
pr = prediction(normTestData, bvalue)
print(binaryPrediciton(pr))

#ac = accuracy(pr, normTestLabels)
TPR, FPR = CalcAccuracy(pr, normTestLabels)
print("TPR: {} \nFPR: {}".format(TPR, FPR))
#print(TruePositive(pr, normTestLabels))
#print(pr)
#print(ac)
#print(TrueNegative(pr, normTestLabels))
#print(TPTNaccuracy(pr, normTestLabels))
#plt.plot(bvalue)
#plt.plot(gradientDescent(normTrainData, normTrainingLabels, bvalue))
#plt.show()