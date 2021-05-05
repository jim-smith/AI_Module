import numpy as np
from random import random, getrandbits




def load_dataset1():
    
    #dataset 1: simple 2 class, 
    #6  features, but only the last has non-zero values in

    numTr = 150
    numFeatures=6
    data1 = np.zeros((numTr,numFeatures))
    labels1 = np.zeros(numTr)
    keyFeature = numFeatures - 1

    data1[0][keyFeature]= 5.00
    labels1[0] = 0
    data1[1][keyFeature]= 4.90
    labels1[1] = 1
    data1[3][keyFeature]= 9.99
    labels1[3] = 0
    data1[4][keyFeature]= 0.0
    labels1[4] = 1
    
    for i in range(4,numTr):
        data1[i][keyFeature] = 4.75*random()
        labels1[i] = 1
        if (bool(getrandbits(1)) ):
            data1[i][keyFeature] +=5.25
            labels1[i]=0
  

    train1_X= data1[0:100,:]
    train1_y= labels1[0:100]
    test1_X= data1[100:,:]
    test1_y= labels1[100:]       
    
    return train1_X,train1_y,test1_X,test1_y


def load_dataset2():
    # dataset 2: 3 classes, first linearly seperable from other two, 
    #other two are once the cases from class 0 ar removed 
    # relevant feature are 2nd and 3rd, all others are pure noise from U(0,100)

    numSamplesPerClass= 50
    numFeatures=6
    data2 = np.random.rand(3*numSamplesPerClass,numFeatures+1)*10 #  one for label 


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
    
    # now random points in this box for the other 138 instances
    for i in range ( 6,3*(numSamplesPerClass ),3):
        # region labelled 0 covers x 6-8, y1-5 
        data2[i][1], data2[i][2], data2[i][numFeatures] = 6.1 +random()*1.8,1.1 + random()*3.8,0
        # region labelled 1 covers x 1-5, y0-2
        data2[i+1][1], data2[i+1][2], data2[i+1][numFeatures] = 1.1 +random()*3.8, random()*1.9, 1
        # region labelled 2 covers x 1-5, y3-5
        data2[i+2][1], data2[i+2][2], data2[i+2][numFeatures] = 1.1 +random()*3.8, 3.1 + random()*1.8, 2
 

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

    #. populate data arrays from U(0,4.5) to start with
    data3 = np.random.rand(n,numFeatures) *4.5
    labels3 = np.zeros(n)

    #select random feature to be key
    keyFeature =  np.random.randint(numFeatures) 
    
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