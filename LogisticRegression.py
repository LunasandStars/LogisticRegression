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

#sigmoid function
#See references above
def sigmoid(value):
    return 1/(1.0 + numpy.exp(-value))


def binaryPrediction(predictionData):
    #array = predictionData
    for label in range(len(predictionData)):
        if predictionData[label] >= 0.5:
            predictionData[label] = 1
        else:
            predictionData[label] = 0
    return predictionData


# Cost function
#See references above
#Sean in class helped me with the logic for the cost function, refering me to the reference above
def cost(X, y, b):
    pr = prediction(X, b)
    C1 = -y * numpy.log(pr)
    C2 = -(1 - y) * numpy.log(1 - pr)
    value = C1 + C2
    return sum(value) / len(y)

# likelihood function
# def likelihood(data, labels, bvalue):
#     scores = numpy.dot(data, bvalue)  # Takes the dot product of the data and bestimate
#     summation = numpy.sum(labels * scores - numpy.log(1 + numpy.exp(scores)))
#     return summation

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

#This calculates the TPR and FPR
#Credit goes to Shah in class for helping me think through the logic of TPR and FPR
def CalcTPRandFPR(predictionLabels, normTestLabels):
    predictionLabels = (predictionLabels > 0.5)
    trueCases = sum(normTestLabels)
    falseCases = len(normTestLabels) - trueCases
    TP, TN, FP, FN = 0, 0, 0, 0
    for x in range(len(predictionLabels)):
        if numpy.logical_and(predictionLabels[x], normTestLabels[x]):
            if predictionLabels[x] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if (predictionLabels[x] != normTestLabels[x]):
                if predictionLabels[x] == 1:
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

TPRArray = [] #y-axis
FPRArray = [] #x-axis
AccuracyArray = []
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
        cost_plot.append(cost(normTrainData, normTrainingLabels, bvalue))


    plt.plot(cost_plot)
    plt.title('Cost Function')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Learning Rate')
    plt.show()

    # # Prediction with Gradient Descent
    pr = prediction(normTestData, bvalue)
    print(binaryPrediction(pr))


    # plt.plot(cost(normTrainingLabels, normTestLabels, bvalue))
    # plt.show()

    ac = accuracy(pr, normTestLabels)
    AccuracyArray.append(ac)
    TPR, FPR = CalcTPRandFPR(pr, normTestLabels)
    TPRArray.append(TPR)
    FPRArray.append(FPR)

print("TPR: {} \nFPR: {}".format(TPR, FPR))
print("Accuracy: {}".format(ac))

TPRArray += [0, 1] #y-axis adding 0 and 1 to the array
FPRArray += [0, 1] #x-axis adding 0 and 1 to the array
TPRArray.sort()
FPRArray.sort()
plt.plot(FPRArray, TPRArray)
plt.title('ROC Curve')
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.show()