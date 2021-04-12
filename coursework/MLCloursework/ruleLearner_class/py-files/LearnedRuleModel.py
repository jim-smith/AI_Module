import numpy as np

class LearnedRuleModel:
    
    def __init__(self,maxRules=5, increments=25):
 
        #read some user-configurable paramerers
        ## maximum size of ruleset that is our model
        self.maxRules= maxRules
        ## number of threshold values to consider for each feature
        self.numThresholds=increments
        
        # initialise currentModel to be an empty array with no rules that scores 0
        self.ruleSet = np.empty((self.maxRules,4),dtype=np.uint)
        self.numRules=0
        self.score=0
    
    def GetRuleSet(self):
        # only return the  rows from the  array that we have actually stored rules in
        return self.ruleSet[:self.numRules,: ]
    
    def PrintRuleSet(self):
        ops = ("<","==",">")
        if (self.numRules==0):
            print("\t Empty Model - No rules learned")
        else:
            print("\tThe Learned Model is: ")
            for rule in range (self.numRules ):
                thisRule = self.ruleSet[rule]
                ruleFeature = thisRule[0]
                ruleOperator = ops[thisRule[1]]
                ruleThreshold= self.thresholds[ruleFeature][thisRule[2]]
                startString = "\t"
                if rule>0:
                    startString = "\tELSE "
                print("{}IF feature {} {} {:.3f} THEN label= {}".
                  format(startString,ruleFeature,ruleOperator,ruleThreshold,thisRule[3]))
            
        
    def fit(self, train_X,train_y):
        # in the super class this will just preprocess the data but not use it
        # so your fit() method should start with these 3 code lines (and their comments) but then do something !
        
        # store the set of different labels - don't assume what they are
        self.labels = np.unique(train_y)

        # remember how many features there are describing each training case
        self.numFeatures= train_X.shape[1]
        
        # preprocess the data to compute the set of thresholds  to be used in rules
        # there are self.numThresholds of these for each feature
        self.CalculatePossibleThresholds(train_X)
        
        # now learn a set of rulea from the training data ..
        
    
    def predict(self,examples):
        # create an empty array to store the predictions in 
        ypred = np.zeros(examples.shape[0])
        for i in range (len(ypred)):
            ypred[i] = -999 #use this value to mean "no label"
        
        # go into a loop making a predeiction for each test case
        for i in range (len(ypred)):
            # In your class you should write  code to  apply your learned rules to each test case
            # to replace the 3 lines of "dumb" code below which cycle through the valid labels
            numLabels = len(self.labels)
            labelIdx = i % numLabels
            ypred[i] = self.labels [labelIdx]
       
        # In your predict function you will need to complete this code 
        # to check and  assign a default class if no rules match
        for i in range (len(ypred)):
            if(ypred[i] == -999):
                pass
                
        # return the predicted label for each test case
        return ypred       
    
 
    def CalculatePossibleThresholds(self, data):
        # method to calculate the set of threshold values to loop over for each feature
        # result is a set of evenly spaced thresholds for each feature that our rule set can refer to
        
        numItems = data.shape[0]
        
        # make an array to hold them: 1 row for each feature, one column for each threshold we will store
        self.thresholds = np.empty((self.numFeatures, self.numThresholds))
        
        # numpy code to get the max and min values for each feature
        maxValues = np.max(data,axis=0)#these are arrays of siuze numFeatures
        minValues = np.min( data, axis=0)
        
        # loop through features
        for thisFeature in range (self.numFeatures):
            # calculate gap between thresholds based on range of values present for this feature
            thisFeatureIncrement = (maxValues[thisFeature] - minValues[thisFeature] )/self.numThresholds
            # step through the thresholds calculating  and  saving their values to an array
            for thresh in range ( self.numThresholds):
                self.thresholds[thisFeature][thresh] = minValues[thisFeature] + thresh * thisFeatureIncrement

                
    def InstanceMatchesRule( self,rule, instance):
        # method to return True:False depending on whether an instance (example) matches the condition of a rule
        
        #interpret the rule and make sure this method is being called correctly
        if len(rule) != 4:
            print("error in ItemMatchesRule, rule must contain exactly four values")
            return False
        
        feature = int(rule[0])
        if( feature <0 or feature> self.numFeatures):
            print('invalid feature id encountered')
            return False
        
        operator = int(rule[1])
        if( operator <0 or operator>2):
            print('invalid operator id encountered')
            return False
        
        thresholdId = int(rule[2])
        if( thresholdId <0 or thresholdId>= self.numThresholds):
            print('invalid threshold index encountered')
            return False
        else:
            threshold = self.thresholds[feature][thresholdId]
        
        # now test the relevant conditions for the rule
        if( (operator==0) and  (instance [feature] < threshold) ): # op is "less than"
            return(True)
        elif (operator ==1 and  ( instance [feature] == threshold) ): #op is "equals"
            return True
        elif ( (operator ==2) and ( instance [feature]> threshold)): # op is "greater than"
            return True
        else:
            return False
        
        
    def Score( self,newRuleSet, numRules, dataset_X,dataset_y):
        # method to take a (partial) model and a dataset_X
        # makes predictions for each training instance
        # returns how many predictions match labels,
        # or returns  -1 if *any* predictions are  incorrect
        # during training, a model is allowed  to not make a prediction for an instance
        numToPredict = dataset_X.shape[0]
        numCorrect=0
        
        noPrediction = -999 # dummy Value to make code easier to read
        
        # loop through each case to make a prediction for
        for instance in range( numToPredict):
            #initialise prediction to a value meaning "NO_PREDICTION"
            prediction = noPrediction 
            
            #now try to apply the model rules until a prediction is made OR run out of rules
            for currentRule in range (numRules):
                # see if case matches this rule's conditions
                if ( self.InstanceMatchesRule(newRuleSet[currentRule], dataset_X[instance] )==True):
                    # if so predict label from rule and go to next case
                    prediction = self.labels[newRuleSet[currentRule][3]]
                    break
                    
            # see if a prediction was made
            if ( prediction != noPrediction):
                if ( prediction == dataset_y[instance] ):
                    numCorrect +=1
                else: # stop scoring as soon as we find incorrect prediction
                    numCorrect = -1
                    break

        return numCorrect
  
        

