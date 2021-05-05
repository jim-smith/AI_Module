## Jim's version that jsut runs the code 100 times and saves, then reports on, the scores
# THIS IS AIMED TO BE QUESTION INDEPENDENT

import student, numpy as np,the_marking as Marking
numRuns = 100

for run in range(numRuns):
    
    mark,output  = Marking.MarkingFunction()
    print("run {} score {}".format(run,mark))
    if(mark!= 100):
        print(output)
       # break

