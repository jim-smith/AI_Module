import LearnedRuleModel
import numpy as np
import load_datasets as ld


from correct_GreedyRuleInductionModel import GreedyRuleInductionModel as Correct
from student import GreedyRuleInductionModel as Student
from simple_1NN_PretendingToBeRuleModel import GreedyRuleInductionModel as KNN


from err_msg import processError, getExceptionDetails

# MARK-A: THE PURPOSE
#
# Runs the student's code on three different datasets
#


##======================================
## def TestAlgorithm on Dataset
# runs a provided model on aprovided dataset:
# this gets called 3 models (correct, student,knn) x 3 datasets times
def TestAlgorithmOnDataset(theModel,X_train, y_train,X_test,y_test,verbose=False):

    debug = verbose #controls level of detail printed during run
    
    theModel.fit(X_train,y_train)
    numStoredRules= theModel.numRules
    if(debug):
        print('\tAfter calling the fit function the model thinks it has {} rules'.format(numStoredRules))
        theModel.PrintRuleSet()

    ypred = theModel.predict(X_test)
    
    # arrays to check no invalid predictions
    classesPresent = np.unique(y_train)
    numClasses = len(classesPresent)
    invalidPreds = []
              
              
    confusionMatrix = np.zeros((numClasses,numClasses+1),np.uint)
    accuracy = 0.0
 
    for i in range(len(y_test)):
        actual = int(y_test[i])
        predicted = int(ypred[i])
        #put all the erroneous predictions into an extra column in the confusion matrix and store them for feedback
        if(predicted not in classesPresent):
            if(debug):
              print("Classifier wrongly predicted a label {} which does not exist in the training set".format(predicted))
            invalidPreds.append(predicted)
            predicted=numClasses
 
        confusionMatrix[actual][predicted] += 1
        if(actual==predicted):
            accuracy += 1.0
        else:
            pass#print("error on item {}, {} actual {} predicted {}".format(i,X_test[i],actual,predicted))

    accuracy = accuracy*100/len(y_test)
    if (len(invalidPreds) >0):
          accuracy = -1
    
    if(debug):
        print("\tTest accuracy on this dataset is {}%, and the confusion matrix is:".format(accuracy))
        header= "\tPredicted|"
        for label in range(numClasses):
            header = header + "  " + str(int(classesPresent[label])) + " "
        header = header + " Invalid"
        print(header)
        for row in range(numClasses):
            rowString= "\tActual   |"
            for column in range(numClasses+1):
                rowString = rowString + "  " + str(confusionMatrix[row][column] )
            print(rowString)
 
    
    return numStoredRules,accuracy, invalidPreds




#==============================
# takes a ruleset from a students model and checks it using the predict function from the correct model
def getCorrectPredictionsForStudentRuleSet(ruleSet,numRules,correctModel, test_X):
    thresholds = correctModel.thresholds
    corr_ypred = np.zeros(test_X.shape[0],dtype=np.uint)
    for item in range (test_X.shape[0]):
        prediction = -1 # i'm going to use -1 to denote "NO_PREDICTION"
        for currentRule in range (numRules):
            if ( correctModel.InstanceMatchesRule(ruleSet[currentRule], test_X[item] )==True):
                prediction = ruleSet[currentRule][3]
                break
        if(prediction == -1): #set to default value if no predicvtion made
            prediction = 0
        corr_ypred[item]= correctModel.labels[prediction]
            
    return corr_ypred



##==================================================
##
### This is the main marking function
##
##
def MarkClassifier(): 

    num_incs = 100 # number of threasholds tested
    
    datasets = [ld.load_dataset1(), ld.load_dataset2(), ld.load_dataset3()]


    setDesc = ("a simple 2-class problem with one predictive variable, all others are zero",
           "a three class problem defined by values for 2 variables, rest are noise",
          "a simple 2 class problem, defined by one variable, where the rest are noise")
    correctDeltas = np.zeros(len(datasets))
    knnDeltas = np.zeros(len(datasets))
    madeInvalid = 0
    madeNoRules = 0
    message = ""
    for setNum in range(len(datasets)):

        #get data
        train_X,train_y,test_X,test_y = datasets[setNum]
 
        #results for correct, knn and s tudent classifiers
        correctModel = Correct(maxRules=5,increments=num_incs)   
        currectNumRules, correctTestScore,_ = TestAlgorithmOnDataset (correctModel,train_X,train_y,test_X,test_y,verbose=False)
        knnModel = KNN(maxRules=5,increments=num_incs) 
        _, KNNTestScore,_  = TestAlgorithmOnDataset (knnModel,train_X,train_y,test_X,test_y,verbose=False)
    
        studentModel = Student(maxRules=5,increments=num_incs) 
        studentNumRules, studentTestScore, studentInvalids = TestAlgorithmOnDataset (studentModel,train_X,train_y,test_X,test_y,verbose=False)
    
        correctDeltas[setNum] = np.abs(correctTestScore - studentTestScore)
        knnDeltas[setNum] = np.abs(KNNTestScore - studentTestScore)
    
        message = message + "On dataset {}, {}, your method got a test accuracy of {}\n".format(setNum,setDesc[setNum],studentTestScore)
        message = message +"\tThe target for this dataset was {} so on this dataset the difference was {}\n".format(correctTestScore,correctDeltas[setNum])

        # feedback about their predict() method from testing the student's learned model with my code
        if(studentNumRules>0):
            message = message + "\tTesting the set of {} rules learned by your code using a different implementation of predict()\n".format(studentNumRules)
            studentRuleSet = studentModel.GetRuleSet()
            corr_ypred = getCorrectPredictionsForStudentRuleSet(studentRuleSet,studentNumRules,correctModel, test_X)
            st_ypred = studentModel.predict(test_X)
            if(np.array_equal(st_ypred, corr_ypred)):
                message = message + "\tThis gave the same results, indicating that  your predict() method is probably correct.\n"
            else:
                 message = message + "\tThis gave a different results indicating that your predict() method is not correct.\n"
        else:# (studentNumRules==0)
            madeNoRules += 1
            message = message + "\tHowever, your algorithm had a value of 0 for self.numRules, and getRuleSet() returned an empty array.\n"
            message = message + "\tSo we could not test your predict() method for this dataset.\n"

        
        if(len(studentInvalids)>0):
            message = message +"\tCrucially, you algorithm predicted labels that were not present in the training set. This should be impossible.\n"
            madeInvalid +=1
        message = message + "\n"
        
               #delete objects
        del knnModel
        del correctModel
        del studentModel
        del train_X,train_y,test_X,test_y

 
    # Finally print summary message and score
    message = message+"<br><span style='color:blue;font-weight:bold'>Overall</span>:\n"
    finalScore = 0
    if (madeInvalid>0):
        message = message + "Your algorithm made some invalid predictions, therefore you score 0"
        finalScore = 0
    else:
        wrongAlg = False
        if(madeNoRules >0 and madeNoRules < len(datasets)):
            message = message + "Your algorithm made  rules on some datasets but not on {}. ".format(madeNoRules)
            message = message + "That indicates you are not generating all possible rules- check your loop conditions."
            
        if(madeNoRules ==len(datasets)):
            message = message + "Your algorithm made no rules for any datasets."
            message = message + " This  suggests you have implemented a different algorithm or not used the super class correctly.\n"
            wrongAlg = True
        if(np.mean(knnDeltas) < 2):
            message = message + "The results are very close to what a 1-NN classifier gets. Did you implement this by mistake?\n"

        meanDiff= np.mean(correctDeltas)
        maxDiff = np.max(correctDeltas)
        message = message + "Your algorithm makes valid predictions."

        if(maxDiff>=10):
            message = message + " However,  the test accuracy more than 10 percent away from the target on one or more datasets, so you score 20\n"
            finalScore = 20
        elif ( meanDiff <10 and meanDiff>5):
            message = message + "The mean difference to the target test accuracy is between 5 and 10 percent, so you score 40.\n"
            finalScore=40
        elif ( (meanDiff <= 5.0)and (meanDiff >2.5) and (maxDiff >= 5)):
            message = message + "The mean difference to the target accuracy is between 2.5 and 5 percent."
            message = message + " However, the difference is greater than 5 for one or more datasets so you score 50.\n"
            finalScore = 50
        elif ( (meanDiff <= 5.0)and (meanDiff >2.5) and (maxDiff <5)):
            message = message + "On average your  test accuracy is between 2.5 and 5 percent away from the target."
            message = message + " The difference is less than 5 for all datasets so you score 60.\n"
            finalScore = 60
        elif(meanDiff >= 1.0 and meanDiff<2.5):

            theMsg = "On average your test accuracy, {:.2f}, is within 1 and 2.5 percent of the target accuracy, so you score 70.\n"
            message += theMsg.format(meanDiff)
            finalScore = 70
        
        elif(meanDiff < 1.0):

            theMsg = "The mean difference to the target test accuracy is {:.2f}, is less than 1 percent, so you score 100 for this run.\n"
            message += theMsg.format(meanDiff)
            finalScore = 100
        
        
    return finalScore,message       
#======================