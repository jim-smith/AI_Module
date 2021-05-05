from LearnedRuleModel import LearnedRuleModel 
import numpy as np

class GreedyRuleInductionModel(LearnedRuleModel):
    
    def __init__(self,maxRules=10, increments=25):
        # call the init function for the super class
        # and inherit all the other methods
        super().__init__(maxRules=maxRules, increments=increments)


        
    def fit( self,train_X,train_y):
       # store the set of different labels - don't assume theyare 0,12,2 etc
        self.labels = np.unique(train_y)

        # remember how many features there are describing each trainig case
        self.numFeatures= train_X.shape[1]
        
        # preprocess the data to compute the set of thresholds  to be used in rules
        # there are self.numThresholds of these for each feature
        self.CalculatePossibleThresholds(train_X)
  
        # now to learn from the data
        
        ###== YOUR CODE HERE====####
    
    def predict(self, examples):
        ypred = np.zeros(examples.shape[0],dtype=np.uint)
 
        ###== YOUR CODE HERE====####
            
        return ypred
    

