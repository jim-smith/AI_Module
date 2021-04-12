## Jim's version that jsut runs the code 100 times and saves, then reports on, the scores
# THIS IS AIMED TO BE QUESTION INDEPENDENT

import student, numpy as np,the_marking as Marking
numRuns = 100
marks = np.zeros(numRuns)
for run in range(numRuns):
    marks[run], _ = Marking.MarkingFunction()
    print ("run {}: {}".format(run, marks[run]))
    
print ( " mean {} min {} max {}".format(marks.mean(),marks.min(),marks.max()))   
