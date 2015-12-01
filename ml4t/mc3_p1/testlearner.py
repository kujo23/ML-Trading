"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import KNNLearner as knn
import BagLearner as bl
import matplotlib.pyplot as plt

if __name__=="__main__":
    inf = open('Data/ripple.csv')
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = math.floor(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    insample_rmse_lin = []
    insample_rmse_knn = []
    outsample_rmse_lin= []
    outsample_rmse_knn = []

    # for i in range(1,50):
    #     # create a learner and train it
    #     # learner = lrl.LinRegLearner() # create a LinRegLearner
    #     learner = knn.KNNLearner(k = i)
    #     # learner = bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":i}, bags = 20, boost = False)
    #     learner.addEvidence(trainX, trainY) # train it

    #     # evaluate in sample
    #     predY = learner.query(trainX) # get the predictions
    #     rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    #     insample_rmse_lin.append(rmse)
    #     print
    #     print "In sample results"
    #     print "RMSE: ", rmse
    #     c = np.corrcoef(predY, y=trainY)
    #     print "corr: ", c[0,1]

    #     # evaluate out of sample
    #     predY = learner.query(testX) # get the predictions
    #     rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    #     outsample_rmse_lin.append(rmse)
    #     print
    #     print "Out of sample results"
    #     print "RMSE: ", rmse
    #     c = np.corrcoef(predY, y=testY)
    #     print "corr: ", c[0,1]

    for i in range(1,100):
        # create a learner and train it
        # learner = lrl.LinRegLearner() # create a LinRegLearner
        # learner = knn.KNNLearner(k = i)
        learner = bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":3}, bags = i, boost = False)
        learner.addEvidence(trainX, trainY) # train it

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        insample_rmse_knn.append(rmse)
        print
        print "In sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=trainY)
        print "corr: ", c[0,1]

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        outsample_rmse_knn.append(rmse)
        print
        print "Out of sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0,1]

    # plt.plot(range(1,50) , insample_rmse_lin, label = "In Sample Error without Bagging", color = "c")
    # plt.plot(range(1,50) , outsample_rmse_lin, label = "Out Sample Error without Bagging", color = "g")

    plt.plot(range(1,100) , insample_rmse_knn, label = "In Sample Error", color = "r")
    plt.plot(range(1,100) , outsample_rmse_knn, label = "Out Sample Error", color = "b")
    plt.xlabel('Bags')
    plt.ylabel('RMSE Error')
    plt.legend(loc = 'upper left')
    plt.show()
