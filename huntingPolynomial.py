import numpy as np
import matplotlib.pyplot as plt
import math
import argparse


__author__ = "Apoorv Sharma"
__email__ = "****"
__studentId__ = "_"


parser = argparse.ArgumentParser(description='Running the PolyHunt Project')
parser.add_argument('-g','--gamma', metavar='', type=float, help='regularization constant (default as 0)')
parser.add_argument('-tp','--trainPath', metavar='', required=True, type=str, help='filepath to the training data')
parser.add_argument('-mo','--modelOutput', metavar='', type=str, help='a filepath where the best fit parameters will be saved')
parser.add_argument('-I','--info', metavar='', action=argparse.BooleanOptionalAction, type=bool, help='if this flag is set, name and contact information will be printed')
parser.add_argument('-nf','--numFolds', metavar='', type=int, help='the number of folds to use for cross validation')
autofitGroup = parser.add_argument_group('autofitGroup')
autofitGroup.add_argument('-A','--autofit', metavar='', action=argparse.BooleanOptionalAction, type=bool, help='flag which when supplied engages the order sweeping loop; if not supplied, fit the polynomial of the given order')
autofitGroup.add_argument('-m', '--order', metavar='', type=int, help='polynomial order (or maximum order in autofit mode)')
parser.set_defaults(gamma=0, autofit=False, info=False)
args = parser.parse_args()


def data_splitter(a):
    x = []
    y = []
    for b in a:
        c1, c2 = b.split(',')
        x.append(float(c1))
        y.append(float(c2))
    return np.array(x), np.array(y)


def generatePhi(x, m):
    matrix = np.zeros((len(x), m+1))
    for power in range(m+1):
        matrix[:, power] = x**power
    return matrix


def linearRegression(x, y, m, phi, gamma):
    phiT = phi.T
    phiTphi = np.dot(phiT, phi)
    regularization = gamma * np.eye(m+1)
    phiTphiINV = np.linalg.inv(phiTphi + regularization)
    model = np.dot(np.dot(phiTphiINV,phiT),y)
    return model


def predictingY(phi, model):
    return np.dot(phi, model)


def calculateError(y_hat, y, gamma, model):
    sqDiff = np.square(np.add(y_hat, (-1) * y))
    sumD = np.sum(sqDiff)
    sumW = np.dot(model.T, model)
    error = 0.5 * sumD + (gamma/2) * sumW
    return math.sqrt((2 * error)/len(y))


def regressing(x_train, y_train, x_test, y_test, order, gamma):
    phi = generatePhi(x_train, order)
    model = linearRegression(x_train, y_train, order, phi, gamma)
    yPred = predictingY(phi, model)
    errorTrain = calculateError(yPred, y_train, gamma, model)
    # print("error for training: ",errorTrain)
    phiTest = generatePhi(x_test, order)
    yPredTest = predictingY(phiTest, model)
    errorTest = calculateError(yPredTest, y_test, gamma, model)
    # print("error for test: ",errorTest)
    return errorTest, model


def huntingPolynomial(m,gamma,trainPath,modelOutput, modelO,autofit,info,numFolds):
    if info:
        print("Name: ",__author__," \nStudent id: ",__studentId__," \nUR Email: ",__email__,"\n")
    x = np.loadtxt(trainPath, dtype=str)
    train, test = np.split(x, [int(.9 * len(x))])
    xTrain, yTrain = data_splitter(train)
    xTest, yTest = data_splitter(test)
    # print("** Autofit: ", autofit)
    model = []
    order = m
    if autofit:
        print("** sweep **")
        errorL = []
        for k in range(1, 30):
            # print("order: ", k)
            error = implementCrossValidation(x, numFolds, k)
            errorL.append(error)
        minE = min(errorL)
        order = errorL.index(minE) + 1
        print("min error among orders with cross validation: ", minE)
        print("order of polynomial: ", order)
        error, model = regressing(xTrain, yTrain, xTest, yTest, order, gamma)
        print("model weights: ",model)

    elif not autofit:
        print("** fit **")
        error, model = regressing(xTrain, yTrain, xTest, yTest, order, gamma)
        print("error: ", error)
        print("model weights: ", model)
    if modelO:
        print("Saving best fit parameters in: ", modelOutput)
        np.savetxt(modelOutput, model, header="m = %d\ngamma = %f" % (order, gamma))


def implementCrossValidation(data,numFolds, k):
    index = range(len(data))
    split = np.array_split(index, numFolds)
    errorF = 0
    for i in range(numFolds):
        # print("performing cross validation for fold: ",i)
        trainL = []
        for j in range(numFolds):
            if i != j:
                trainL.extend(split[j])
        xTest, yTest = data_splitter(data[split[i]])
        xTrain, yTrain = data_splitter(data[trainL])
        error, model = regressing(xTrain, yTrain, xTest, yTest, k, gamma)
        errorF = errorF + error
    return errorF/len(data)


if __name__ == '__main__':
    m = args.order
    gamma = args.gamma
    trainPath = args.trainPath
    numFolds = args.numFolds
    # trainPath = 'A'
    modelO = False
    modelOutput = args.modelOutput
    autofit = args.autofit
    if autofit == 'None':
        autofit = False
    info = args.info
    if info == 'None':
        info = False
    if modelOutput is not None:
        modelO = True
    huntingPolynomial(m, gamma, trainPath, modelOutput, modelO, autofit, info, numFolds)
