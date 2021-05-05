from LearnedRuleModel import LearnedRuleModel  
import numpy as np

class GreedyRuleInductionModel(LearnedRuleModel):
    
    def __init__(self,maxRules=10,increments=10):
        # this version only looks at the single nearest neighbour
        super().__init__(maxRules=maxRules, increments=increments)
        
    def fit(self,train_X,train_y):
        
        # store the set of different labels - don't assume theyare 0,12,2 etc
        self.labels = np.unique(train_y)

        # remember how many features there are describing each trainig case
        self.numFeatures= train_X.shape[1]
        
        # preprocess the data to compute the set of thresholds  to be used in rules
        # there are self.numThresholds of these for each feature
        self.CalculatePossibleThresholds(train_X)
        
        
        
        # ask the data how big it is and store that info
        self.numExemplars = train_X.shape[0]
        self.numFeatures = train_X.shape[1]
        # store a copy of the data (X) and the labels (y)
        self.modelX = train_X
        self.modelY = train_y
        self.labelsPresent = np.unique(self.modelY) # list the unique values found in the labels provided
  
    def predict(self,newItems):
        # see how many  newitems there are
        numToPredict = newItems.shape[0]
        # make an empty list to hold their predicted labels
        predictions = np.empty(numToPredict)
        
        #loop through each new item each one
        for item in range(numToPredict):
            # predicting its label
            thisPrediction = self.PredictNewItem ( newItems[item])
            # adding that predictin to our list
            predictions[item] = thisPrediction
        return predictions
    
    def PredictNewItem(self,newItem):
        
        # Step 1: measure and store distance to each training item
        distFromNewItem = np.zeros((self.numExemplars)) # array with one entry for each trainig set item, intialised to zero
        for exemplar in range (self.numExemplars):
            distFromNewItem[exemplar] = self.EuclideanDistance(newItem,  self.modelX[exemplar])
  
        # Step 2: find the one closest training example: This is K=1, 
        closest = 0
        for trainingExample in range (0, self.numExemplars):
            if  ( distFromNewItem[trainingExample] < distFromNewItem[closest] ):
                closest=trainingExample
 
        # step 3: count the votes - because this is for K=1 so we don't need to take a vote
        labelOfClosest = self.modelY[closest]
        return labelOfClosest
    
    def EuclideanDistance(self,a,b):
        ## this numpy function calculates the euclidean distance
        return np.linalg.norm(a-b)    


