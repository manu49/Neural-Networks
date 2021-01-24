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



############################################################################## SCIKIT PART ######################################################################

y_pred = clf.predict(test_data)
cf_scikit = np.zeros((10,10))

i = 0
while(i<len(test_labels)):
    cf_scikit[int(y_pred[i])][int(test_labels[i])] += 1
    i=i+1




cf_cvxopt = np.zeros((10,10))


############################################################################# CONFUSION MATRICES ##################################################################




class SupportVecMachine(object):

    def __init__(self):
        
        self.C = float(1)

    def make_P(self,m,y,X):
        temp = np.zeros((m,m))

        i = 0

        while(i<m):
            j = 0
            while(j<m):
                temp[i,j] = np.exp((-linalg.norm(X[i]-X[j])**2) *0.05 )
                j=j+1
            i=i+1


        raw_P = np.outer(y,y) * temp
        P = cvxopt.matrix(raw_P)

        return(P)



    def make_A(self,y,m):
        y1 = np.array([float(elem) for elem in y])

        A = cvxopt.matrix(y1, (1,m))
        return(A)


    def make_h(self,m):
        tmp1 = np.zeros(m)
        tmp2 = np.ones(m) * float(1)
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        return(h)

    def make_G(self,m):
        tmp1 = np.diag(np.ones(m) * -1)
        tmp2 = np.identity(m)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        return(G)


    def fit(self, X, y):
        m = X.shape[0]
        n = X.shape[1]

        
        
        P = self.make_P(m,y,X)
        A = self.make_A(y,m)
        G = self.make_G(m)
        h = self.make_h(m)
        print("p done")

        raw_q = (-1) * np.ones(m)
        q = cvxopt.matrix(raw_q)
        print("q done")

        


        #A = A.astype('float')
        print(len(A))
        print("a done")
        #prin

        b = cvxopt.matrix(0.0)

        
    
        
        

        print("params calculated")
        solution = solvers.qp(P,q,G,h,A,b)

        print("solved")
        print(solution['primal objective'])

        
        a = np.ravel(solution['x'])

        
        sv = a > 1e-5

        ind = np.arange(len(a))[sv]

        self.a = a[sv]

        self.vectorsx = X[sv]
        self.vectorsy = y[sv]

        self.num_of_support_vectors = len(self.a)

        print("num of support vectors = "+str(len(self.a))+" "+str(self.num_of_support_vectors))

        
        self.b = 0
        #self.w = np.zeros(n)


        temp = np.zeros((m,m))

        i = 0

        while(i<m):
            j = 0
            while(j<m):
                temp[i,j] = np.exp((-linalg.norm(X[i]-X[j])**2) *0.05 )
                j=j+1
            i=i+1


        n1 = 0
        while(n1 < self.num_of_support_vectors):
            self.b += self.vectorsy[n1]
            self.b -= np.sum(self.a * self.vectorsy * temp[ind[n1],sv])
            n1 = n1 + 1

        self.b /= self.num_of_support_vectors

        
        
        self.w = None
        

    def bs(self):
        return(self.b)

    def ws(self):
        return(self.w)

    def nosv(self):
        return(self.num_of_support_vectors)


    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            lk = len(X)
            y_predict = np.zeros(lk)

            tuples = zip(self.a, self.vectorsy, self.vectorsx)
            i = 0
            while(i<lk):
                s = 0
                for a, sv_y, sv in tuples:
                    expo = (-1)*linalg.norm(X[i]-sv)**2
                    s += a * sv_y * np.exp(expo*0.05 ) ## gaussian kernel
                y_predict[i] = s
                i+=1
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))





def make_dict():
    dicti = {}
    i = 0
    while(i<10):
        dicti[i] = 0
        i=i+1

    return(dicti)



def get_dat(num):
    train_data_loc = sys.argv[1]
    test_data_loc = sys.argv[2]
    output_loc = sys.argv[3]


    #print(train_data_loc)
    print("train : " + train_data_loc)
    print("test : " + test_data_loc)

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
            if(lab==num or lab==((num+1)%10)):
                temp = [float(elem) for elem in temp]
                raw_train_data.append(np.array(temp))
                if(lab==num):
                    raw_train_labels.append(-1)
                else:
                    raw_train_labels.append(1)
                

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
            if(lab == num or lab == ((num+1)%10)):
                temp = [float(elem) for elem in temp]
                raw_test_data.append(np.array(temp))
                if(lab==num):
                    raw_test_labels.append(-1)
                else:
                    raw_test_labels.append(1)

    #raw_test_data = [float(elem) for elem in raw_test_data]
    test_data = np.array(raw_test_data)*(1/255)
    test_labels = np.array(raw_test_labels)

    print("data done")

    return(train_data,train_labels,test_data,test_labels)







def get_dat1(num1,num2):
    train_data_loc = sys.argv[1]
    #test_data_loc = sys.argv[2]
    #output_loc = sys.argv[3]


    #print(train_data_loc)
    print("train : " + train_data_loc)
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
            if(lab==num1 or lab==num2):
                temp = [float(elem) for elem in temp]
                raw_train_data.append(np.array(temp))
                if(lab==num1):
                    raw_train_labels.append(-1)
                else:
                    raw_train_labels.append(1)
                

    #raw_train_data = [float(elem) for elem in raw_train_data]
    train_data = np.array(raw_train_data)*(1/255)
    train_labels = np.array(raw_train_labels)


    '''raw_test_data = []
    raw_test_labels = []

    with open(test_data_loc, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            p = len(row) - 1
            temp = row[0:p]
            lab = int(float(row[p]))
            if(lab == num or lab == ((num+1)%10)):
                temp = [float(elem) for elem in temp]
                raw_test_data.append(np.array(temp))
                if(lab==num):
                    raw_test_labels.append(-1)
                else:
                    raw_test_labels.append(1)

    #raw_test_data = [float(elem) for elem in raw_test_data]
    test_data = np.array(raw_test_data)*(1/255)
    test_labels = np.array(raw_test_labels)

    print("data done")'''

    return(train_data,train_labels)


def get_dat2():
    #train_data_loc = sys.argv[1]
    test_data_loc = sys.argv[2]
    #output_loc = sys.argv[3]


    #print(train_data_loc)
    #print("train : " + train_data_loc)
    #print("test : " + test_data_loc)

    '''raw_train_data = list([])
    raw_train_labels = list([])

    with open(train_data_loc, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            #print(row)
            p = len(row)-1
            temp = row[0:p]
            lab = int(float(row[p]))

            #print(type(lab))
            if(lab==num or lab==((num+1)%10)):
                temp = [float(elem) for elem in temp]
                raw_train_data.append(np.array(temp))
                if(lab==num):
                    raw_train_labels.append(-1)
                else:
                    raw_train_labels.append(1)
                

    #raw_train_data = [float(elem) for elem in raw_train_data]
    train_data = np.array(raw_train_data)*(1/255)
    train_labels = np.array(raw_train_labels)'''


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

    return(test_data,test_labels)













def test_soft(num1,num2,d,X_test):
    '''X1, y1, X2, y2 = gen_lin_separable_overlap_data()
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)'''

    

    X_train,y_train = get_dat1(num1,num2)

    clf = SupportVecMachine()
    #print("initialised")
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)

    #print("prediction done")
    #correct = np.sum(y_predict == y_test)
    '''y = 0
    for t in y_predict:
        if(t==1):
            y += 1'''

    #print("%d out of %d predictions correct" % (correct, len(y_predict)))
    '''if(y_predict[0]==-1):
        d[(num1)] += 1
    else:
        d[num2] += 1'''

    t = 0
    while(t<len(X_test)):
        if(y_predict[t]==1):
            d[t][num1] += 1
        else:
            d[t][num2] += 1
        t += 1





def make_dict1(test_dat):
    l = len(test_dat)
    d = {}
    i = 0
    for f in test_dat:
        d[i] = np.zeros(10)
        i = i + 1

    return(d)








def final_test(cm):

    test_data1, test_labels1 = get_dat2()
    

    prediction_labels = np.zeros(len(test_labels1))

    t = 0


    main_dictionary = make_dict1(test_data1)


    i = 0
    
    while(i<=9):
        j=i+1
        while(j<=9):
            test_soft(i,j,main_dictionary,test_data1)
            j+=1
        i+=1

    


    t = 0
    for entry in test_data1[:1]:

        #d = make_dict()
        i = 0
        '''while(i<=9):
            test_soft(i,d,np.array(test_data1[i]))
            print("iteration "+str(i)+" done")
            i=i+1'''

        j = 0
        idx = 0
        m = 0
        while(j<10):
            if(m<main_dictionary[t][j]):
                m=main_dictionary[t][j]
                idx = j
            j=j+1

        prediction_labels[t] = idx
        t = t + 1


    correct = np.sum(prediction_labels == test_labels1)

    #print("accuracy = "+ str(correct/len(test_labels1)))


    i = 0
    while(i<len(test_labels1)):
        cm[int(prediction_labels[i])][int(test_labels1[i])] += 1
        i+=1

    #print(len(test_labels1))

    return(cm)


######################################################### CVXOPT PART #################################################################3



#cf_scikit = np.zeros((10,10))
cf_cvxopt = final_test(cf_cvxopt)






file1 = open(output_loc,"w")

m1 = np.matrix(cf_scikit)
m2 = np.matrix(cf_cvxopt)

file1.write("For scikit learn module : \n")
for m11 in m1:
    np.savetxt(file1, m11, fmt='%.2f')


file1.write("\n For cvxopt module : \n")
for m22 in m2:
    np.savetxt(file1,m22,fmt='%.2f')


file1.close()



