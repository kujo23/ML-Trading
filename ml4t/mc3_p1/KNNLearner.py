import numpy as np
import math

class KNNLearner(object):

    def __init__(self, k):
        self.dataX = None
        self.dataY = None
        self.k = k

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.dataX = dataX.copy()
        self.dataY = dataY.copy()
        
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        retY = []

        for point in points:
            euclid_dist = np.array([np.linalg.norm(point - data) for data in self.dataX])
            indices = np.argsort(euclid_dist)
            meanY = np.mean(self.dataY[indices[0:self.k]])
            retY.append(meanY)

        return retY
        
if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
