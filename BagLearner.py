"""
A simple wrapper for Bootstrap Aggregating.  (c) 2017 BAOFENG ZHANG
"""

import numpy as np
from random import *
import RTLearner as rt

class BagLearner(object):
    
    

    def __init__(self,learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 10, boost = False, verbose = False):
        #pass # move along, these aren't the drones you're looking for
        self.learners = []
        
        for i in xrange(bags):
            self.learners.append(learner(**kwargs))
        
    def author(self):
        return 'bzhang367' 
        
        
    def split_point(self,dataX):
        
        index = []
        for i in xrange(int(0.6 * dataX.shape[0])):
            index.append(randint(0, dataX.shape[0] - 1))
        return index
        

    def addEvidence(self, dataX, dataY):
        
        for learner in self.learners:
            index = self.split_point(dataX)
            trainX = []
            trainY = []
            for i in index:
                trainX.append(dataX[i])
                trainY.append(dataY[i])
            learner.addEvidence(np.array(trainX),np.array(trainY))
                   
    def query(self,points):
        
        Ypredict = None
        for learner in self.learners:
            
            R = learner.query(points)
            if Ypredict is None:
                Ypredict = R
            else:
                Ypredict = np.add(Ypredict, R)
        
        Ypredict = Ypredict / len(self.learners)
        return Ypredict
        

if __name__=="__main__":
    print "wonderful job"
