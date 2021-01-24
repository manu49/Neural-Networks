######## done


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



def make_partitions(n,data,labels):
	l = len(labels)
	s = l/n
	st = 0
	en = s 
	partitions = []
	while(en <= l and st<en):
		x = data[st:en]
		y = labels[st:en]
		partitions.append([x,y])
		st = st+s
		en = min(l,en+s)

	return(partitions)



def split(i,data,labels,sz):
	test_data_ = data[i:(i+sz)]
	test_labels_ = labels[i:(i+sz)]
	train_data_ = np.vstack((data[:i],data[(i+sz):]))
	train_labels_ = np.hstack((labels[:i],labels[(i+sz):]))

	return(test_data_,test_labels_,train_data_,train_labels_)



X = train_data
Y = train_labels


k = 5
bs = int(len(train_labels)/k)
c_array = [1e-5,1e-3,1,5,10]


idx1 = 0
idx2 = 0
m1 = 0
m2 = 0

i = 0
while(i<k):
	c = c_array[i]
	print("using c = "+str(c))
	accuracies1 = np.zeros(k)
	accuracies2 = np.zeros(k)
	j = 0
	while(j<k):

		print("iter : "+str(j))
		ts,tl,tr,trl = split(j,train_data,train_labels,bs)
		clf = svm.SVC(kernel='rbf',gamma=0.05,C=c)
		clf.fit(tr,trl)
		accuracies1[j] = clf.score(ts,tl)
		accuracies2[j] = clf.score(test_data,test_labels)
		j=j+1



	acc1 = np.mean(accuracies1)


	acc2 = np.mean(accuracies2)

	if(acc1>m1):
		m1 = acc1
		idx1 = i
	if(acc2 > m2):
		m2 = acc2
		idx2 = i 
	print("for iteration : "+str(i))
	print(acc1)
	print(acc2)

	i=i+1

#clf = svm.SVC(kernel = 'rbf', random_state = 0, gamma = 0.05, C=1)
#clf.fit(X, Y)

#acc = clf.score(test_data,test_labels)


file1 = open(sys.argv[3],"w")

file1.write("best val set : "+str(c_array[idx1])+" with accuracy = "+str(m1*100)+" percent\n")
file1.write("best test set : "+str(c_array[idx2])+" with accuracy = "+str(m2*100)+" percent\n")


#print(acc)

et = time()


file1.write("total time taken = "+str(et-st)+" seconds\n")

file1.close()










'''

y_pred = clf.predict(test_data)
cf_scikit = np.zeros((10,10))

i = 0
while(i<len(test_labels)):
    cf_scikit[y_pred[i]][test_labels[i]] += 1
    i=i+1




cf_cvxopt = np.zeros((10,10))

'''