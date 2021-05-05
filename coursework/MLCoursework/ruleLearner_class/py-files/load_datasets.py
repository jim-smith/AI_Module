import numpy as np
from random import random, getrandbits



def load_dataset1():
    
    #dataset 1: simple 2 class, 
    #6  features, but only the last has non-zero values in
    n = 150
    numFeatures=6
    keyFeature = numFeatures - 1
    

    #. populate data arrays with zeros to start with
    # and key feature from U(0,4.5)
    data1 = np.zeros((n,numFeatures))
    for i in range(n):
        data1[i][keyFeature] = np.random.rand() *4.5
    labels1 = np.zeros(n)

 
    
    #make sure training set has example at desired boundaries
    data1[0][keyFeature] = 5.00
    labels1[0] = 1
    data1[1][keyFeature] = 4.90
    labels1[1] = 0
    data1[2][keyFeature] = 9.99
    labels1[2] = 1
    data1[3][keyFeature] = 0.0
    labels1[3] = 0
    # then apply a simple 2-class one feature labelling scheme
    for i in range(4,n):
        data1[i][keyFeature] = 4.75*random()
        if (bool(getrandbits(1)) ):
            data1[i][keyFeature] +=5.25
            labels1[i]=1


    train1_X= data1[0:100,:]
    train1_y= labels1[0:100]
    test1_X= data1[100:,:]
    test1_y= labels1[100:]       
    
    return train1_X,train1_y,test1_X,test1_y


def load_dataset2():
    # dataset 2: 3 classes, first linearly seperable from other two, 
    # other two are seperable once the cases from class 0 ar removed 
    # relevant feature are 1 and 2, 
    # features 3 and 4 are pure noise to foll kNN
    # features  0 and 5 are  zeros

    n= 150
    numFeatures=6
    data2 = np.zeros ((n,numFeatures+1)) 
    
    randFeatures = [3,4]
    for feat in randFeatures:
        for i in range (n):
            data2[i][feat] = np.random.rand()*100 #  one for label 


    #make the data in triples
    #starting by providing points at the bounding box edges so
    # I know min and max x-y values are present in the training set

    # region labelled 0 covers x 6-8, y1-5 
    data2[0][1] , data2[0][2], data2[0][numFeatures] = 6,1,0
    data2[1][1], data2[1][2], data2[1][numFeatures] = 6,5,0
    data2[2][1] , data2[2][2], data2[2][numFeatures] = 8,1,0
    data2[3][1], data2[3][2], data2[3][numFeatures] = 8,5,0
    # region labelled 1 covers x 1-5, y0-2
    data2[4][1], data2[4][2], data2[4][numFeatures] = 1,0,1
    data2[5][1], data2[5][2], data2[5][numFeatures] = 1,2,1
    data2[6][1], data2[6][2], data2[6][numFeatures] = 5,0,1
    data2[7][1], data2[7][2], data2[7][numFeatures] = 5,2,1
    # region labelled 2 covers x 1-5, y3-5
    data2[8][1], data2[8][2], data2[8][numFeatures] = 1,3,2
    data2[9][1], data2[9][2], data2[9][numFeatures] = 1,5,2
    data2[10][1], data2[10][2], data2[10][numFeatures] = 5,3,2
    data2[11][1], data2[11][2], data2[11][numFeatures] = 5,5,2

    #print(data2[:6, :])
    
    # now random points in this box for the  other training instances
    for i in range ( 6,100,3):
        # region labelled 0 covers x 6-8, y1-5 
        data2[i][1], data2[i][2], data2[i][numFeatures] = 6 +random()*2,1.0 + random()*45,0
        # region labelled 1 covers x 1-5, y0-2
        data2[i+1][1], data2[i+1][2], data2[i+1][numFeatures] = 1.0 +random()*4, random()*2, 1
        # region labelled 2 covers x 1-5, y3-5
        data2[i+2][1], data2[i+2][2], data2[i+2][numFeatures] = 1.0 +random()*4, 3 + random()*2, 2
    # slightly smaller box for test instances
    for i in range ( 99,n,3):
        # region labelled 0 covers x 6-8, y1-5 
        data2[i][1], data2[i][2], data2[i][numFeatures] = 6.25 +random()*1.5,1.25 + random()*3.5,0
        # region labelled 1 covers x 1-5, y0-2
        data2[i+1][1], data2[i+1][2], data2[i+1][numFeatures] = 1.25 +random()*3.5, random()*1.75, 1
        # region labelled 2 covers x 1-5, y3-5
        data2[i+2][1], data2[i+2][2], data2[i+2][numFeatures] = 1.25 +random()*3.5, 3.25 + random()*1.5, 2
 

    #splt data into train & test, shuffle rows
    train= data2[0:100,:]
    test = data2[100:,:]
    #shuffle rows
    np.random.shuffle(train)
    np.random.shuffle(test)
    #split into X and y
    train2_X= train[: , :numFeatures]
    train2_y= train[: , numFeatures]
    test2_X=   test[: , :numFeatures]
    test2_y=   test[: , numFeatures]
    
    

    return(train2_X,train2_y,test2_X,test2_y)


def load_dataset3():
    #dataset 3: simple 2 class, 
    #6  features with uniformly distributed values,
    # only 1 randomly chosen feature is important

    from random import random
    n = 150
    numFeatures=6

    #. populate data arrays with mix of zeros for first and last columns, others from from U(50) to start with
    data3 = np.random.rand(n,numFeatures) *50
    for i in range(n):
        data3[i][0],data3[i][numFeatures-1] = 0.0,0.0
    labels3 = np.zeros(n)

    #select random feature to be key
    keyFeature =  1  + np.random.randint(numFeatures -2) 
    
    #make sure training set has example at desired boundaries
    data3[0][keyFeature] = 5.00
    labels3[0] = 1
    data3[1][keyFeature] = 4.90
    labels3[1] = 0
    data3[2][keyFeature] = 9.99
    labels3[2] = 1
    data3[3][keyFeature] = 0.0
    labels3[3] = 0
    # then apply a simple 2-class one feature labelling scheme
    for i in range(4,n):
        data3[i][keyFeature] = 4.75*random()
        if (bool(getrandbits(1)) ):
            data3[i][keyFeature] +=5.25
            labels3[i]=1


    train3_X= data3[0:100,:]
    train3_y= labels3[0:100]
    test3_X= data3[100:,:]
    test3_y= labels3[100:]
    
    return( train3_X,train3_y,test3_X,test3_y)