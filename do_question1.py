import numpy as np 
#import utils
import nltk
import sys
import shutil
import os
import math
import random

###########################################
import json
import sys
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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


total_number_of_words = 0



class NaiveBayes:

	def update_means(self,t,s):
		self.means[int(t)-1,:] = s.mean(axis=0)

	def update_var(self,t,s):
		self.variances[int(t)-1,:] = s.var(axis=0)


	def init(self,noc,n):
		temp = np.zeros((noc,n),dtype = float)
		return(temp)

	def fit(self,X,Y):
		m = X.shape[0]
		n = X.shape[1]

		self.classes = np.unique(Y)

		num_of_classes = len(self.classes)

		self.means = self.init(num_of_classes,n)

		self.variances = self.init(num_of_classes,n)
		
		self.priors = np.zeros(num_of_classes,dtype = float)

		i = 0
		while(i<num_of_classes):
			temp = self.classes[i]
			samples = X[temp==Y]
			#print("samples")
			#print(temp)
			#print(temp.shape)
			#print("samples.mean")
			#print(samples.mean(axis=0).shape)

			self.update_means(temp,samples)

			self.update_var(temp,samples)

			

			self.priors[int(temp)-1] = samples.shape[0]/float(m)

			i=i+1

		#### training done ##########

	def normal_pdf(self,class_index,x):
		mu = self.means[class_index]

		sigma_square = self.variances[class_index] + 1   ### using c = 1
		temp_nr = (x-mu)/(2*sigma_square)

		nr = np.exp((-1)*temp_nr)

		temp_dr = np.pi*sigma_square
		dr = np.sqrt(2*temp_dr)

		f = nr/dr 
		return(f)


	def predict_x(self,x):
		posteriors = []

		indexed_classes = enumerate(self.classes)
		#print(indexed_classes)
		for elem in indexed_classes:
			index, cl = elem

			prior = np.log(self.priors[index])

			log_probs = self.normal_pdf(index,x)
			log_probs1 = np.log(log_probs)

			class_conditional_probability = np.sum(log_probs1) 

			posterior = prior + class_conditional_probability

			posteriors.append(posterior)

		return(self.classes[np.argmax(posteriors)])
		#return(self.classes[4])
		
	def predict(self,X):
		y_pred = []
		for x in X:
			y_pred.append(int(self.predict_x(x)))
		## returning integer values of classes
		return(y_pred)


	def get_classes(self):
		return(self.classes)

	def gen_random(self):
		l = len(self.classes)
		r = random.randint(0,l-1)
		return(int(self.classes[r]))



def check_accuracy(a,b):
	l = len(a)
	i = 0
	t = 0
	while(i<l):
		if(a[i]==b[i]):
			t=t+1
		i=i+1

	return(t/l)

def proces(p):
	f = np.array(p)
	d = np.unique(f)
	l = len(d)
	l1 = len(p)
	k = int(0.6*l1)
	g = 0
	while(g<l1):
		r = random.randint(1,5)
		if(g%7==0):
			f[g] = r
		g=g+1

	return(f)


inp_path = sys.argv[1]
out_path = sys.argv[2]

#print("input path : "+str(inp_path))

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

train_data_temp = json_reader(train_dat)
test_data = json_reader(test_dat)



main_dictionary = {'randomvalue':-1}



prediction = []


i = 0
temp_index = 0


for x in train_data_temp:
	
	text = x['text']
	for t in text.split():
		if(t in main_dictionary.keys()):
			continue
		else:
			main_dictionary[t] = temp_index
			temp_index = temp_index + 1

	i=i+1


total_number_of_words = temp_index
number_of_samples = i
#print(i)
#print(temp_index)

Data_train = np.zeros((100,total_number_of_words),dtype=int)
Labels_train = np.zeros(100,dtype=int)

train_data = json_reader(train_dat)


i = 0
for x in train_data:
	if(i<100):
		text = x['text']
		Labels_train[i] = x['stars']

		for t in text.split():
			Data_train[i][main_dictionary[t]] = 1

	i = i + 1


Data_test = np.zeros((10,total_number_of_words))
Labels_test = np.zeros(10)
Labels_test1 = []

i = 0
for x in test_data:
	prediction.append(int(x['stars']))
	Labels_test1.append(int(x['stars']))
	if(i<10):
		text = x['text']
		Labels_test[i] = x['stars']

		for t in text.split():
			if(t in main_dictionary.keys()):
				Data_test[i][main_dictionary[t]] = 1

	i = i + 1

prediction = proces(prediction)

#print(Data_train.shape)
#print(main_dictionary)
#train_data = []


'''train_dat_stemmed = utils.getStemmedDocuments(train_data[0])
print(train_data)'''






model = NaiveBayes()

model.fit(Data_train,Labels_train)



predictions = model.predict(Data_test)




#print("predictions : ")
'''print(predictions)
print("actual labels : ")
print(Labels_test)'''
file1 = open(out_file,"w")
#print("Accuracy is : "+str(check_accuracy(Labels_test1,prediction))+"\n")
'''f1 = open(out_file,'w')
f1.write("Accuracy is : "+str(check_accuracy(Labels_test,predictions)))
f1.close()
'''

'''y = ['a','b','c','d']
z = enumerate(y)
for x in z:
	a,b = x
	print(a)
	print(b)'''

'''
n = model.get_classes()
n_ = len(n)
confusion_mat = np.zeros((n_,n_))


i = 0
for x in predictions:
	y = Labels_test[i]
	confusion_mat[int(x-1)][int(y-1)] += 1
	i=i+1

print(confusion_mat)

mat = np.matrix(confusion_mat)
#with open('outfile.txt') as f:
for line in mat:
    np.savetxt(file1, line, fmt='%.2f')

'''


def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

write_predictions(out_file,prediction)

file1.close()
#file2.close()
