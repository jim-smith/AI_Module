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
              
              
        #WHILE (currentModel.score<trainingsetSize) DO 
        while (self.score<numTrItems and self.numRules<self.maxRules):
            #print("\tAdding to model with {} rules which score {}".format(self.numRules,self.score))
            # SET bestchild = emptyModel
            bestChild= self.ruleSet.copy()
            bestChildScore=self.score
   
            
            #FOR newRule in  (all_possible_rules)
            for feature in range(self.numFeatures):
                for operator in range (3):
                    for threshold in range (self.numThresholds):
                        for label in range(len(self.labels)):
                            newRule= np.array ( [feature,operator,threshold,label])
                            #   SET newModel = COPY(currentModel)
                            newModel = self.ruleSet.copy()
                            ##  SET newModel = ADDRULE (newModel, newRule)
                            newModel[self.numRules]= newRule
                            #  SET score = SCORE(newModel)
                            newScore = self.Score(newModel, self.numRules+1,train_X,train_y)
                            #  IF (newModel.score > bestChild.score)
                            if (newScore>bestChildScore):
                                #SET bestChild= COPY(newModel)
                                bestChild = newModel.copy()
                                bestChildScore = newScore
                                #print("new best rule for this iteration classifies {}".format(bestChildScore))

            #IF (bestChild.score > currentModel.score)
            if (bestChildScore > self.score):
                #SET currentModel=COPY (bestChild)
                self.ruleSet = bestChild.copy()
                self.numRules += 1
                self.score = bestChildScore
            else:
                #print("\t...exiting training loop becuase no improving rule could be found")
                break

                
        #print("\tModel has {} rules and a training set accuracy of {}".format(self.numRules,(100*self.score/numTrItems)))


    
    def predict(self, examples):
        ypred = np.zeros(examples.shape[0],dtype=np.uint)
 
        for item in range (examples.shape[0]):
            prediction = -1 # i'm going to use -1 to denote "NO_PREDICTION"
            for currentRule in range (self.numRules):
                if ( self.InstanceMatchesRule(self.ruleSet[currentRule], examples[item] )==True):
                    prediction = self.ruleSet[currentRule][3]
                    break
            if(prediction == -1): #set to default value if no predicvtion made
                prediction = 0
            ypred[item]= self.labels[prediction]
            
        return ypred
    

