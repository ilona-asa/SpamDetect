import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_iris
import os
import re
import time

from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import cross_validation
from sklearn.neural_network import MLPClassifier
from sklearn import cross_validation

# ## Part 3: Reading a text-based dataset into pandas

# In[38]:

# read file into pandas using a relative path
rootdir = '/root/Desktop/Machine_Learning/Project-SpamDetection/enron1/'
tic = time.clock()
toc = time.clock()
toc - tic
#rootdir = 'preprocessed_data/enron1/'
listtexts_spam = []
listtexts_ham = []
labels_spam = []
labels_ham = []
listtexts = []
labels = []
spam = 0
ham = 0
for subdirs,dir,files in os.walk(rootdir):
    for file in files:
        path =  os.path.join(subdirs, file)
        if '.idea' in path:
            continue
        elif 'py' in path:
            continue
        elif 'Summary.txt' in path:
            continue
        else:
            f = open(path)
            email = f.read()
            em = email.splitlines()

            Text = ""

            for e in em:
                #if "Subject:" in e:
                if re.search(r'\bSubject:', e):
                    e = e[9:]
                Text = Text + " " + e
        #Text = Text.encode('utf-8').strip()
        Text = re.sub(r'[^\x00-\x7F]+',' ', Text)
        listtexts.append(Text)

        if 'spam' in subdirs:
            labels.append('Spam')
        else:
            labels.append('Ham')
"""        if 'spam' in subdirs:
            labels_spam.append('Spam')
            listtexts_spam.append(Text)
            spam += 1
        else:
            labels_ham.append('Ham')
            listtexts_ham.append(Text)
            ham += 1
"""

#print len(labels_spam)
#print len(labels_ham)
#print(labels)

#print len(listtexts_spam)
#print len(listtexts_ham)

#randomize all the dataset for ham and spam separately

"""ham_text_random, ham_labels_random = shuffle(listtexts_ham, labels_ham, random_state=0)
spam_text_random, spam_labels_random = shuffle(listtexts_spam, labels_spam, random_state=0)
all_text = ham_text_random + spam_text_random
"""

#split dataset into train set and test set
"""def splitData(ham_text, spam_text, ham_label, spam_label, split):
    ham_split = int(split*len(ham_text))
    print(ham_split)
    spam_split = int(split*len(spam_text))
    print(spam_split)
    train_set_text = ham_text[:ham_split] + spam_text[:spam_split]
    train_set_label = ham_label[:ham_split] + spam_label[:spam_split]
    test_set_text = ham_text[ham_split:] + spam_text[spam_split:]
    test_set_label = ham_label[ham_split:] + spam_label[spam_split:]
    return train_set_text, train_set_label, test_set_text, test_set_label
"""

from sklearn.utils import shuffle
listtexts_random, labels_random = shuffle(listtexts, labels, random_state=0)

def splitData(textdata, label, trainSize):
    split = int(trainSize*len(textdata))
    train_set_text = textdata[:split]
    train_set_label = label[:split]
    test_set_text = textdata[split:]
    test_set_label = label[split:]
    return train_set_text, train_set_label, test_set_text, test_set_label

#train_set_text, train_set_label, test_set_text, test_set_label = splitData(listtexts_random, labels_random, 0.7)

##Vectorized the data
vect = CountVectorizer(stop_words='english')
# learn the 'vocabulary' of the training data (occurs in-place)
vect.fit(listtexts_random)

# examine the fitted vocabulary
features_name = vect.get_feature_names()
print (len(features_name))	
#print(type(features))
#print(len(features))

# transform training data into a 'document-term matrix'
simple_train_dtm = vect.transform(listtexts_random)

# convert sparse matrix to a dense matrix
features_matr = simple_train_dtm.toarray()

# examine the vocabulary and document-term matrix together
import pandas as pd
features = pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
print(len(features))

featuresnumpy = features.as_matrix()

##split data
train_set_text, train_set_label, test_set_text, test_set_label = splitData(features, labels_random, 0.7)

#***************Start kNN ***************************
tic = time.clock()
# instantiate the model (with the default parameters)
knn = KNeighborsClassifier()
print knn
# fit the model with data (occurs in-place)
knn.fit(train_set_text, train_set_label)
predict_knn = knn.predict(test_set_text)
toc = time.clock()
elapsed_knn = toc - tic
print("Classification report for classifier %s: \n %s \n"
% ('k-NearestNeighbour', metrics.classification_report(test_set_label, predict_knn)))
print("Confusion matrix:\n %s" % metrics.confusion_matrix(test_set_label, predict_knn))
print("k-NN Accuracy score:\n %s" % metrics.accuracy_score(test_set_label, predict_knn))
print "k-NN elapsed time: ", elapsed_knn
kFold = 10
scores = cross_validation.cross_val_score(knn,featuresnumpy,labels_random,cv=kFold)
print(scores)

#***************End kNN ***************************

#**************Start of SVM*******************************************
from sklearn.svm import LinearSVC
tic = time.clock()
clf_svm = LinearSVC()
print clf_svm
clf_svm.fit(train_set_text, train_set_label)
predictedLabels = clf_svm.predict(test_set_text)
toc = time.clock()
acc_svm = metrics.accuracy_score(test_set_label, predictedLabels)
elapsed_SVM = toc - tic
print("Classification report for classifier %s: \n %s \n"
% ('k-NearestNeighbour', metrics.classification_report(test_set_label, predictedLabels)))
print("Confusion matrix:\n %s" % metrics.confusion_matrix(test_set_label,predictedLabels))
print "Linear SVM accuracy: ", acc_svm
print "Linear SVM elapsed time: ", elapsed_SVM
kFold = 10
scores = cross_validation.cross_val_score(clf_svm,featuresnumpy,labels_random,cv=kFold)
print(scores)
#***************End SVM ***************************


#***************Start Naive Bayes Classifier *******************
#instantiate Multinomail naive Bayes model
tic = time.clock()
nb = MultinomialNB()
print nb

#train the model using train_set_text
nbtrain = nb.fit(train_set_text, train_set_label)

#make class predictions for test_set_text
test_pred_class = nb.predict(test_set_text)
toc = time.clock()
elapsed_NB = toc - tic
# calculate accuracy of class predictions
print "NB accuracy"
print metrics.accuracy_score(test_set_label, test_pred_class)

# print the confusion matrix
print "NB confusion matrix"
print metrics.confusion_matrix(test_set_label, test_pred_class)

#classification report
print "NB classification report"
print metrics.classification_report(test_set_label, test_pred_class)
print "Naive Bayes elapsed time: ", elapsed_NB
kFold = 10
scores = cross_validation.cross_val_score(nb,featuresnumpy,labels_random,cv=kFold)
print(scores)

#***************End Naive Bayes Classifier ***************************

#**************Start of MLP*******************************************
tic = time.clock()
clf = MLPClassifier(solver = 'lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state = 1)
#print train_set_label
train_set_label_int = []
for x in train_set_label:
	if x == 'Ham':
		train_set_label_int.append(1)
	else:
		train_set_label_int.append(0)
#print train_set_label_int

clf.fit(train_set_text,train_set_label_int)
predictedLabels = clf.predict(test_set_text)
toc = time.clock()
elapsed_MLP = toc - tic
test_set_label_int = []
for x in test_set_label:
	if x == 'Ham':
		test_set_label_int.append(1)
	else:
		test_set_label_int.append(0)
#print train_set_label_int
print("Classification report for classifier %s: \n %s \n"
% ('MLP', metrics.classification_report(test_set_label_int, predictedLabels)))
print("Confusion matrix:\n %s" % metrics.confusion_matrix(test_set_label_int, predictedLabels))
acc_svm = metrics.accuracy_score(test_set_label_int, predictedLabels)
print "Linear MLP accuracy: ", acc_svm
print "MLP elapsed time: ", elapsed_MLP
kFold = 10
scores = cross_validation.cross_val_score(clf,featuresnumpy,labels_random ,cv=kFold)
print(scores)

#******************End of MLP******************************************
