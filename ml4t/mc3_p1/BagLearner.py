import numpy as np
import LinRegLearner as lrl
import KNNLearner as knn
import math

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost):

        self.dataX = None
        self.dataY = None
        learners = []
        for i in range(0,bags):
            learners.append(learner(**kwargs))
        self.learners = learners
        self.boost = boost


    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.dataX = dataX
        self.dataY = dataY
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        store = np.zeros([len(points),len(self.learners)])

        for i in range(0,len(self.learners)):
            indices = np.random.choice(len(self.dataX),len(self.dataX), replace = True)
            valueX = np.array([self.dataX[index] for index in indices])
            valueY = np.array([self.dataY[index] for index in indices])
            self.learners[i].addEvidence(valueX, valueY)
            store[:,i] = self.learners[i].query(points)

        return np.mean(store,axis = 1)
            
        

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"