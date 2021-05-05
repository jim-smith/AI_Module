


class GreedyRuleInductionModel(LearnedRuleModel):
    
    def __init__(self,maxRules=10, increments=25):
        # call the init function for the super class
        # and inherit all the other methods
        super().__init__(maxRules=maxRules, increments=increments)


        
    def fit( self,train_X,train_y):
      #  Preprocess (trainingset)  
      # store the set of different labels - don't assume theyare 0,12,2 etc
        self.labels = np.unique(train_y)

        # remember how many features there are describing each trainig case
        self.numFeatures= train_X.shape[1]
        
        # preprocess the data to compute the set of thresholds  to be used in rules
        # there are self.numThresholds of these for each feature
        self.CalculatePossibleThresholds(train_X)
  
        # now to learn from the data
        
        ###== YOUR CODE HERE====####
            ## I suggest you copy in the pseudocode from the lecture then code to that
            ## some of the lines in that pseudocode have been covered above
            ## and some are covered in the init() method
        

        
    
    def predict(self, examples):
        ypred = np.zeros(examples.shape[0],dtype=np.uint)
 
   
  
        ###== YOUR CODE HERE====####
            ## I suggest you start by setting everything in ypred to a valid default value
            ## i.e. something from the list self.labels
            
            ## then you will find it useful to look at the score() method in the super class
            ## to see how you can try to match rules from your ruleset  for each  test instance in turn
        
        return ypred
    


