####### done

import numpy as np 
#import utils
import nltk
import sys

import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
###########################################
import json
import sys
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from time import time
__author__= 'KD'


def json_writer(data, fname):
    """
        Write multiple json files
        Args:
            data: list(dict): list of dictionaries to be written as json
            fname: str: output file name
    """
    with open(fname, mode="w") as fp:
        for line in data:
            json.dump(line, fp)
            fp.write("\n")


def json_reader(fname):
    """
        Read multiple json files
        Args:
            fname: str: input file
        Returns:
            generator: iterator over documents 
    """
    for line in open(fname, mode="r"):
        yield json.loads(line)


def _stem(doc, p_stemmer, en_stop, return_tokens):
    tokens = word_tokenize(doc.lower())
    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stemmed_tokens = map(lambda token: p_stemmer.stem(token), stopped_tokens)
    if not return_tokens:
        return ' '.join(stemmed_tokens)
    return list(stemmed_tokens)


def getStemmedDocuments(docs, return_tokens=True):
    """
        Args:
            docs: str/list(str): document or list of documents that need to be processed
            return_tokens: bool: return a re-joined string or tokens
        Returns:
            str/list(str): processed document or list of processed documents
        Example: 
            new_text = "It is important to by very pythonly while you are pythoning with python.
                All pythoners have pythoned poorly at least once."
            print(getStemmedDocuments(new_text))
        Reference: https://pythonprogramming.net/stemming-nltk-tutorial/
    """
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    if isinstance(docs, list):
        output_docs = []
        for item in docs:
            output_docs.append(_stem(item, p_stemmer, en_stop, return_tokens))
        return output_docs
###########################################


train_dat = sys.argv[1]
test_dat = sys.argv[2]

out_file = sys.argv[3]

#file1 = open(train_dat,"r")
#file2 = open(test_dat,"w")
'''s1 = inp_path + '/utils.py'
new_utils = 'utils1.py'

s_file = open(s1,'r')
text = s_file.read()
#print(text)
s_file.close()

t_file = open(new_utils,'w')
t_file.write(text)
t_file.close()'''

#shutil.copyfile(s1,new_utils)
#os.system('cp ' +s1+ ' ' + new_utils)
#import utils1


st1 = time()

train_data_temp = json_reader(train_dat)
test_data = json_reader(test_dat)



main_dictionary = {'randomvalue':-1}






i = 0
temp_index = 0


for x in train_data_temp:
	
	text = x['text']
	words_list = getStemmedDocuments([text])[0]
	words_list = list(set(words_list))
	#print(len(words_list))
	'''if(i==0 or i==1):
		print(words_list)
		print(words_list[1])'''
	for t in words_list:
		#print(t)
		if(t in main_dictionary.keys()):
			continue
		else:
			main_dictionary[t] = temp_index
			temp_index = temp_index + 1

	#print(i)
	i=i+1


total_number_of_words = temp_index
number_of_samples = i


#print(i)
#print(temp_index)

Data_train = np.zeros((200,total_number_of_words))
Labels_train = np.zeros(200)

train_data = json_reader(train_dat)


i = 0
for x in train_data:
	if(i<200):
		text = x['text']
		Labels_train[i] = x['stars']
		for t in getStemmedDocuments([text])[0]:
			Data_train[i][main_dictionary[t]] = 1

	i = i + 1


Data_test = np.zeros((20,total_number_of_words))
Labels_test = np.zeros(20)



#print("Data obtained...")

i = 0
for x in test_data:
	if(i<20):
		text = x['text']
		Labels_test[i] = x['stars']

		for t in getStemmedDocuments([text])[0]:
			if(t in main_dictionary.keys()):
				Data_test[i][main_dictionary[t]] = 1

	i = i + 1


#print("gaussian naive bayes started...")

gnb = GaussianNB()
y_pred = gnb.fit(Data_train, Labels_train).predict(Data_test)

et1 = time()


num_correct = (Labels_test == y_pred).sum()
print(str(num_correct/len(Labels_train)))
acc1 = num_correct/len(Labels_train)

########################################################################  Bernoulli Naive Bayes part ##############################################################


from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st = time()

#print("svm started...")

clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
clf.fit(Data_train,Labels_train)
score = clf.score(Data_test,Labels_test)

et = time()

file1 = open(out_file,"w")
file1.write("Accuracy from Naive Bayes Algorithm using sklearn tools = "+str(acc1*100)+" %\n")
file1.write("Time taken = "+str(et1-st1)+" seconds\n")
file1.write("Accuracy from SVM using Liblinear = "+str(score*100)+" %\n")
file1.write("Time taken = "+str(et-st)+" seconds\n")

file1.close()

###################################################################### SVM Part using Liblinear ##################################################################

