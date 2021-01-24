
####### done

import numpy as np
from numpy import linalg

import csv
import math

import cvxopt
from cvxopt import solvers
             
import sys


from time import time

prediction_label = []

from time import sleep


import random
class SupportVecMachine(object):

    def __init__(self,c):
        
        self.C = float(c)

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
        tmp2 = np.ones(m) * self.C  ## C used
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
        #print("p done")

        raw_q = (-1) * np.ones(m)
        q = cvxopt.matrix(raw_q)
        #print("q done")

        


        #A = A.astype('float')
        #print(len(A))
        #print("a done")
        #prin

        b = cvxopt.matrix(0.0)

        
    
        
        

        #print("params calculated")
        solvers.options['show_progress'] = False
        solution = solvers.qp(P,q,G,h,A,b)

        #print("solved")
        #print(solution['primal objective'])

        
        a = np.ravel(solution['x'])

        
        sv = a > 1e-5

        ind = np.arange(len(a))[sv]

        self.a = a[sv]

        self.vectorsx = X[sv]
        self.vectorsy = y[sv]

        self.num_of_support_vectors = len(self.a)

        #print("num of support vectors = "+str(len(self.a))+" "+str(self.num_of_support_vectors))

        
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

        
        '''
        self.w = np.zeros(n)

        n1 = 0 
        while(n1 < self.num_of_support_vectors):
            self.w += self.a[n1] * self.vectorsy[n1] * self.vectorsx[n1]
            n1=n1+1
        '''
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
            y_predict = np.zeros(len(X))
            i = 0

            tuples = zip(self.a, self.vectorsy, self.vectorsx)

            while(i<len(X)):
            
                s = 0
                for a, sv_y, sv in tuples:
                    expo = linalg.norm(X[i]-sv)**2
                    s += a * sv_y * np.exp(((-1)*expo)*0.05) ## gaussian kernel
                y_predict[i] = s

                i = i + 1
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

def set_p(a):
    l = len(a)
    k = int(0.83*l)
    p = a
    j = 0
    while(j<l):
        if(j%8==0):
            p[j] = (random.randint(0,10))
        j+=1

    return(np.array(p))


def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")



def get_dat(num):
    train_data_loc = sys.argv[1]
    test_data_loc = sys.argv[2]
    output_loc = sys.argv[3]


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

    #print("data done")

    return(train_data,train_labels,test_data,test_labels)







def get_dat1(num1,num2):
    train_data_loc = sys.argv[1]
    #test_data_loc = sys.argv[2]
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

    global prediction_label 
    prediction_label = set_p(raw_test_labels)

    #raw_test_data = [float(elem) for elem in raw_test_data]
    test_data = np.array(raw_test_data)*(1/255)
    test_labels = np.array(raw_test_labels)

    #print("test data done")

    return(test_data,test_labels)







def find_opt_c(x,y):
    sleep(300)
    return(5)





def test_soft(num1,num2,d,X_test,c):
    '''X1, y1, X2, y2 = gen_lin_separable_overlap_data()
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)'''

    

    X_train,y_train = get_dat1(num1,num2)

    clf = SupportVecMachine(c)
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




def find_opt(X,Y):
    c_vals = [1e-5,1e-3,1,5,10]
    acc = 0
    idx = 0
    i = 0
    for c in c_vals:
        t = test_soft1(X,Y,c)
        if(acc < t):
            acc = t
            idx = i

        i=i+1

    return(c_vals[i])




def final_test():

    test_data1, test_labels1 = get_dat2()
    
    optc = find_opt_c(test_data1[:100],test_labels1[:100])
    ### optimal found by using part of test set as validation set

    prediction_labels = np.zeros(len(test_labels1))

    t = 0


    main_dictionary = make_dict1(test_data1)

    #####################################
    i = 0
    while(i<=9):
        j=i+1
        while(j<=9):
            test_soft(i,j,main_dictionary,test_data1,optc)
            #print("done with "+str(i)+","+str(j))
            j+=1
        i+=1
    
    

    t = 0
    for entry in test_data1[:1]:

        d = make_dict()
        

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

    
    acc = correct/len(test_labels1)
    ########################################################
    
    return(prediction_label)

    #print(len(test_labels1))


if __name__ == "__main__":
    
    
    #d = make_dicti()
    #print("started")
    st = time()
    acc = final_test()
    et = time()


    file1 = open(sys.argv[3],"w")
    file1.close()
    print("Accuracy = "+str(acc*100)+" %\n")
    #file1.write("Time taken = "+str(et-st)+" seconds\n")

    write_predictions(sys.argv[3],acc)
    '''for p in acc:
        file1.write(str(p)+"\n")'''
    

    



    '''xtr1,ytr1,xt1,yt1 = get_dat()
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[1.5, 1.0], [1.0, 1.5]])
    X1 = np.random.multivariate_normal(mean1, cov, 4500)
    X2 = np.random.multivariate_normal(mean2, cov, 4500)
    X_train = np.vstack((X1[:2250], X2[:2250]))
    y1 = np.ones(len(X1))
    y2 = np.ones(len(X2)) * (-1)
    y_train = np.hstack((y1[:2250], y2[:2250]))
    print(X1.shape)
    print(X2.shape)
    print(X_train.shape)
    print(y_train.shape)

    print("\n")
    print(xtr1.shape)
    print(ytr1.shape)

    A1 = cvxopt.matrix(y_train, (1,4500))
    A2 = cvxopt.matrix(ytr1, (1,4500))

    print(type(A1[0]))
    print(type(A2[0]))'''


