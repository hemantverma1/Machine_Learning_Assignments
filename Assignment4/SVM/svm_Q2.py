import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import svm
import csv

def checkAccuracy(kernelType):
    C = [1, 10, 100, 1000, 10000]
    D = [2, 3, 4]
    for c in C:
        for d in D:
            accuracy = numpy.array([],dtype=float)
            clf = svm.SVC(kernel=kernelType, degree=d, C=c)
            for i in range(30):
                kf = KFold(n_splits=10,shuffle=True)
                tmpAccuracy = 0.0
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    Label_train, Label_test = Label[train_index], Label[test_index]
                    clf.fit(X_train, Label_train)
                    outputTest = clf.predict(X_test)
                    outputDiff = numpy.subtract(outputTest, Label_test)
                    tmpAccuracy += len(outputDiff) - numpy.count_nonzero(outputDiff)
                tmpAccuracy /= len(Label)
                accuracy = numpy.append(accuracy, tmpAccuracy)
            standardDev = numpy.std(accuracy)
            accuracy = numpy.mean(accuracy)
            print(c, d, accuracy, standardDev)

def plot(kernelType):
    c = 10000
    d = 2
    if kernelType == "poly":
        clf = svm.SVC(kernel=kernelType, degree=d, C=c)
    elif kernelType == "rbf":
        clf = svm.SVC(kernel=kernelType, gamma=d, C=c)
    clf.fit(X, Label)
    h = .002
    # create a mesh to plot in
    x_min = -1.5
    x_max = 1.5
    y_min = -1.5
    y_max = 1.5
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
                         numpy.arange(y_min, y_max, h))
    Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    #plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.pcolormesh(xx, yy, Z > 0, cmap=plt.cm.Paired)
    plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],levels=[-.5, 0, .5])
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Label, marker='o', cmap=plt.cm.coolwarm)
    plt.xlabel('v1')
    plt.ylabel('v2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

if __name__ == "__main__":
    Label = numpy.array([], dtype=int)
    X = numpy.empty((0,2), float)
    with open('Data_SVM.csv','r') as fHandle:
        reader = csv.reader(fHandle)
        next(reader)
        for row in reader:
            Label = numpy.append(Label, (int(row[-1])))
            row = row[:-1]
            X = numpy.vstack((X, numpy.asarray([float(x) for x in row])))
    print("Polynomial Kernel")
    checkAccuracy('poly')
    print("Plotting graph for C=10000, d=2")
    plot('poly')
    print("RBF Kernel")
    checkAccuracy('rbf')
    print("Plotting graph for C=10000, gamma=2")
    plot('rbf')
