####### done

import numpy as np
from numpy import linalg

import csv
import math

import cvxopt
from cvxopt import solvers
             
import sys


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets

from time import time


st = time()




def get_dat2():
    train_data_loc = sys.argv[1]
    test_data_loc = sys.argv[2]
    #output_loc = sys.argv[3]


    #print(train_data_loc)
    #print("train : " + train_data_loc)
    #print("test : " + test_data_loc)

    raw_train_data = list([])
    raw_train_labels = list([])

    with open(train_data_loc, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            #print(row)
            p = len(row)-1
            temp = row[0:p]
            lab = int(float(row[p]))

            #print(type(lab))
            if(lab >= 0):
                temp = [float(elem) for elem in temp]
                raw_train_data.append(np.array(temp))
                raw_train_labels.append(lab)
                

    #raw_train_data = [float(elem) for elem in raw_train_data]
    train_data = np.array(raw_train_data)*(1/255)
    train_labels = np.array(raw_train_labels)


    raw_test_data = []
    raw_test_labels = []

    with open(test_data_loc, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            p = len(row) - 1
            temp = row[0:p]
            lab = int(float(row[p]))
            if(lab >= 0):
                temp = [float(elem) for elem in temp]
                raw_test_data.append(np.array(temp))
                raw_test_labels.append(lab)

    #raw_test_data = [float(elem) for elem in raw_test_data]
    test_data = np.array(raw_test_data)*(1/255)
    test_labels = np.array(raw_test_labels)

    #print("test data done")

    return(train_data,train_labels,test_data,test_labels)





train_data, train_labels, test_data, test_labels = get_dat2()




X = train_data
Y = train_labels
clf = svm.SVC(kernel = 'rbf', random_state = 0, gamma = 0.05, C=1)
clf.fit(X, Y)

acc = clf.score(test_data,test_labels)

#print(acc)

et = time()


output_loc = sys.argv[3]

file1 = open(output_loc,"w")

file1.write("accuracy = "+str(acc*100)+" percent\n")
file1.write("total time taken = "+str(et-st)+"\n")
file1.write("number of points classified = "+str(len(train_labels))+"\n")

file1.close()
#SVC(decision_function_shape='ovo')
'''dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes: 4*3/2 = 6
#6
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes'''
#4