# file workbook6_utilitiers.py
# author Jim Smith 2021
# hods ots of supporting code for the workbook so that the students don't have to be exposed
# to python that is not directly AI related

# of course we feel it would be nice if some of them chose to look ast how things are done ...


import numpy as np
import matplotlib.pyplot as plt


def show_scatterplot_matrix(X,y,featureNames,title=None):
    f = X.shape[1]
    if(len(y) != X.shape[0]):
        print("Error,   the y array  must have the same length as there are rows in X")
        return
    fig, ax = plt.subplots(f,f,figsize=(12,12))
    plt.set_cmap('jet')
    for feature1 in range(f):
        ax[feature1,0].set_ylabel( featureNames[feature1])
        ax[0,feature1].set_xlabel( featureNames[feature1])
        ax[0,feature1].xaxis.set_label_position('top') 
        for feature2 in range(f):
            xdata = X[:,feature1]
            ydata = X[:,feature2]
            ax[feature1, feature2].scatter(xdata,ydata,c=y)
    if title != None:
        fig.suptitle(title,fontsize=16,y=0.925)
        
        
# simple function - currently only works for 2D data - but could easily be extended
def PlotDecisionSurface(trainX,trainy,theClassifier,theTitle,featureNames,xvar=0,yvar=1,stepSize=2.0,minZero=False):
    #create and prettify the plot
    cmap="Set3"
    fig,ax= plt.subplots(figsize=(8, 8))
    ax.set_title(theTitle)
    ax.set_xlabel(featureNames[xvar])
    ax.set_ylabel(featureNames[yvar])

    #define a grid we use to plot the decision boundaries
      #get max/min values for gri edges
    columnMax,columnMin = np.max(trainX,axis=0), np.min(trainX,axis=0)
    if(minZero==True):
        x_min , y_min= 0,0
    else:
        x_min, y_min = columnMin[ xvar]*0.95, columnMin[yvar]*0.95
    x_max, y_max = columnMax[xvar]*1.05, columnMax[yvar]*1.05 
    #make the grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, stepSize),np.arange(y_min, y_max, stepSize))

    #predict and plotfor evey point on the grid
    Z = theClassifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z,cmap=cmap)

    # Plot also the training points
    ax.scatter(x=trainX[:,xvar ],y= trainX[:, yvar], c=trainy.astype(float), alpha=1.0, cmap=cmap, edgecolor="black")
            
        

        
        
################# the student marks dataset ################

def load_student_marks_dataset(datafilepath="../lectures/data/assessment-grades-2features.csv"):
    grades= np.genfromtxt(datafilepath, delimiter= ',',skip_header=1)

    featureNames=("exam", "CW_mean")
    outcomes= ("Pass","Resit Exam", "Resit Coursework","Resit Both")
    simpleoutcomes= ("pass","resit")

    nStudents = grades.shape[0]


    # make target labels
    result = np.empty(nStudents, dtype=np.int8)

    for row in range (nStudents):
        exam = grades[row][0]
        cw   = grades[row][1]
        if (exam>=35 and cw>=35 and (exam +cw >=80) ):
            result[row] = 0 # PASS 

        elif ( cw>=40 and exam < 40):
            result[row] = 1 #resit just exam 
        elif ( cw<40 and exam>=40):
            result[row]= 2 # resit just coursework
        else:
            result[row] = 3  # resit both
        
    simpleResult = np.where(result<1,0,1)

    return grades, result, simpleResult




def plot_student_marks(grades,result,simpleResult):
    nStudents = grades.shape[0]
    passStudents = np.empty((0,2))
    resitCWStudents = np.empty((0,2))
    resitExamStudents = np.empty((0,2))
    resitBothStudents = np.empty((0,2))

    for student in range (nStudents):
        if (result[student]==0):
            passStudents = np.vstack( (passStudents,grades[student]) )
        elif (result[student]==1):
            resitExamStudents = np.vstack( (resitExamStudents,grades[student]) )
        elif (result[student]==2):
            resitCWStudents = np.vstack( (resitCWStudents,grades[student]) )
        else:
            resitBothStudents = np.vstack( (resitBothStudents,grades[student]) )
    simpleResitStudents = np.vstack( (resitExamStudents,resitCWStudents,resitBothStudents))

    print("Using the simple labelling schme, {} students passed the module and {} had to retake the exam, coursework, or both"
           .format(passStudents.shape[0], simpleResitStudents.shape[0]))
        
    print( "Using the more complex labelling scheme,  those resit broke down into {} exam only, {} coursework only, and {} both"
          .format(resitExamStudents.shape[0], resitCWStudents.shape[0],resitBothStudents.shape[0]))
    
    fig,ax = plt.subplots(1,2,figsize=(14,5))
    plt.xlabel("Exam")
    plt.ylabel("Coursework")
    ax[0].set_title("Outcomes")
    ax[1].set_title("Simplified Outcomes")

    ax[0].scatter(passStudents[:,0],passStudents[:,1],label = "Pass" )
    ax[0].scatter(resitExamStudents[:,0],resitExamStudents[:,1],label = "Resit Exam" )
    ax[0].scatter(resitCWStudents[:,0],resitCWStudents[:,1],label = "Resit CW" )
    ax[0].scatter(resitBothStudents[:,0],resitBothStudents[:,1],label = "Resit Both" )
    ax[1].scatter(passStudents[:,0],passStudents[:,1],label = "Resit" )
    ax[1].scatter(simpleResitStudents[:,0],simpleResitStudents[:,1],label = "Pass" )

    ax[0].legend(loc='lower right')
    ax[1].legend(loc='lower right') 
    
    