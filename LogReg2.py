import numpy
from sklearn.model_selection import KFold
import sklearn as sk
#print(sk.__name__)
import matplotlib.pyplot as plt

#REFERENCES: https://beckernick.github.io/logistic-regression-from-scratch/
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.show.html
#https://docs.eyesopen.com/toolkits/cookbook/python/plotting/roc.html
#http://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#cost-function

# Normalize the data knowing the values are from 0 to 255
def normalizeData(dataNorm):
    return dataNorm / 255.0

#If label = 6 then 0, if label = 8 then 1
#Class 1: 6 and Class 2: 8
def makeBinary(data):  #data = labels
    zeroArray = numpy.zeros(len(data))
    for label in range(len(data)):
        if data[label] == 8:
            zeroArray[label] = 1
    return zeroArray


#Gradient Descent
def gradientDescent(X, y, b):
    #reference https://github.com/michelucci/Logistic-Regression-Explained/blob/master/MNIST%20with%20Logistic%20Regression%20from%20scratch.ipynb
    predictions = prediction(X, b)
    sample = X.shape[0]
    # error_cost = predictions - y
    error_cost = (predictions - y).transpose()
    # return  numpy.dot(X.transpose(), predictions - y)
    return -1.0 / sample * numpy.dot(X.transpose(), error_cost)

#Counts the number of true data
def trueCount(data):
    return sum(data)
#
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

def binaryPrediciton(predictionData):
    #array = predictionData
    for label in range(len(predictionData)):
        if predictionData[label] >= 0.5:
            predictionData[label] = 1
        else:
            predictionData[label] = 0
    return predictionData

#Plot ROC Curve
# def ROCCurve(TPR, FPR, predictionData, randomline = True):
#     plt.figure()
#     plt.xlabel("False Positive Rate", fontsize = 16)
#     plt.ylabel("True Positive Rate", fontsize = 16)
#     plt.title("ROV Curve", fontsize = 16)
#
#     TPR, FPR = CalcAccuracy(predictionData, normTestLabels)
#
#     plt.plot(FPR, TPR, normTestLabels=normTestLabels)
#
#     if randomline:
#         x = [0.0, 1.0]
#         plt.plot(x, x)
#
#     plt.xlim(0.0, 1.0)
#     plt.ylim(0.0, 1.0)

# def addZeroOne(rates):
#     # Cost function
def cost(X, y, b):
    pr = prediction(X, b)
    C1 = -y * numpy.log(pr)
    C2 = -(1 - y) * numpy.log(1 - pr)
    value = C1 + C2
    return sum(value) / len(y)

# likelihood function
def likelihood(data, labels, bvalue):
    scores = numpy.dot(data, bvalue)  # Takes the dot product of the data and bestimate
    summation = numpy.sum(labels * scores - numpy.log(1 + numpy.exp(scores)))
    return summation

# This function makes the prediction using sigmond
def prediction(X, b):
    return sigmoid(numpy.dot(X, b))

# print(prediction(trainingData, b_opt(trainingData, trainingLabels)))

# Classify the test data with threshold of 0.5 (Accuracy)
def accuracy(bestFitLine, testLabels):
    bestFitLine[bestFitLine > 0.5] = 1
    bestFitLine[bestFitLine <= 0.5] = 0
    value = sum(bestFitLine == testLabels) / float(len(testLabels))
    return value

# Accuracy = (TP+TN)/(TP+FP+FN+TN)
def TPTNaccuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + FP + FN + TN)

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


data = numpy.genfromtxt('MNIST_CVHW3.csv', delimiter=',', dtype=int, skip_header=1)


kf = KFold(n_splits = 10)
#print(kf)

print("Starting KFold")

#Split the data with KFold
for train, test in kf.split(data):
    Xytest = numpy.array(data[test])   #data
    Xytraining = numpy.array(data[train])    #Labels


    testingLabels = numpy.array(Xytest[:, 0])
    testingData = numpy.array(Xytest[:, 1: -1])
    trainingLabels = numpy.array(Xytraining[:, 0])
    trainingData = numpy.array(Xytraining[:, 1: -1])


    # print(trainingLabels)
    # Normalize Dataset and Labels
    normTestData = normalizeData(testingData)
    normTrainData = normalizeData(trainingData)
    normTestLabels = makeBinary(testingLabels)
    normTrainingLabels = makeBinary(trainingLabels)

    # print(normTrainData[0])
    # print(normTrainingLabels)


    print("Starting Grad Decent")
    # Getting the b value (weights)
    bvalue = numpy.zeros(len(normTrainData[0]))  # weights
    learningRate = 1e-2
    iterator = 1000
    cost_plot = []
    for x in range(0, iterator):
        bvalue = bvalue + learningRate * gradientDescent(normTrainData, normTrainingLabels, bvalue)
        cost_plot.append( cost(normTrainData, normTrainingLabels, bvalue)  )


    plt.plot(cost_plot)
    plt.show()

    # # Prediction with Gradient Descent
    pr = prediction(normTestData, bvalue)
    print(binaryPrediciton(pr))


    # plt.plot(cost(normTrainingLabels, normTestLabels, bvalue))
    # plt.show()

    #ac = accuracy(pr, normTestLabels)
    TPR, FPR = CalcAccuracy(pr, normTestLabels)
print("TPR: {} \nFPR: {}".format(TPR, FPR))
# plt.plot(ROCCurve(TPR, FPR))
# plt.show()