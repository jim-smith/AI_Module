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
        numTrItems=train_X.shape[0]
  
        #print(" There are {} labels in the data: {}".format(len(self.labels),self.labels))
        #print(" There are {} cases and {} features for each one".format(numTrItems,self.numFeatures))
        #print(" The set of {} thresholds calculated for each feature are:  ".format(self.numThresholds))
        #print(self.thresholds)
              
              
        self.ruleSet[0] = [0,0,0,0]
        self.numRules= 1
                
        #print("\tModel has {} rules and a training set accuracy of {}".format(self.numRules,(100*self.score/numTrItems)))


    
    def predict(self, examples):
        ypred = np.zeros(examples.shape[0],dtype=np.uint)
 
        for item in range (examples.shape[0]):
            prediction = 5
            ypred[item]= prediction#self.labels[prediction]
            
        return ypred
    

