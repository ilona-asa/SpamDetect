import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_iris
import os
import re

# ## Part 3: Reading a text-based dataset into pandas

# In[38]:

# read file into pandas using a relative path
#rootdir = '/root/Desktop/Machine_Learning/Project-SpamDetection/'
rootdir = 'preprocessed_data/'
listtexts_spam = []
listtexts_ham = []
labels_spam = []
labels_ham = []
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
        
        if 'spam' in subdirs:
            labels_spam.append('Spam')
            listtexts_spam.append(Text)
            spam += 1
        else:
            labels_ham.append('Ham')
            listtexts_ham.append(Text)
            ham += 1

print len(labels_spam)
print len(labels_ham)
#print(labels)

print len(listtexts_spam)
print len(listtexts_ham)

#randomize all the dataset for ham and spam separately
from sklearn.utils import shuffle
ham_text_random, ham_labels_random = shuffle(listtexts_ham, labels_ham, random_state=0)
spam_text_random, spam_labels_random = shuffle(listtexts_spam, labels_spam, random_state=0)


#split dataset into train set and test set
def splitData(ham_text, spam_text, ham_label, spam_label, split):
    ham_split = int(split*len(ham_text))
    print(ham_split)
    spam_split = int(split*len(spam_text))
    print(spam_split)
    train_set_text = ham_text[:ham_split] + spam_text[:spam_split]
    train_set_label = ham_label[:ham_split] + spam_label[:spam_split]
    test_set_text = ham_text[ham_split:] + spam_text[spam_split:]
    test_set_label = ham_label[ham_split:] + spam_label[spam_split:]
    return train_set_text, train_set_label, test_set_text, test_set_label

train_set_text, train_set_label, test_set_text, test_set_label = splitData(ham_text_random, spam_text_random, ham_labels_random, spam_labels_random, 0.7)
all_text = train_set_text + test_set_text

##work on the training data
vect = CountVectorizer(stop_words='english')
# learn the 'vocabulary' of the training data (occurs in-place)
vect.fit(all_text)


# In[35]:

# examine the fitted vocabulary
features = vect.get_feature_names()
#print(type(features))
print(len(features))


# In[36]:

# transform training data into a 'document-term matrix'
simple_train_dtm = vect.transform(train_set_text)
#print(type(simple_train_dtm))
#print(simple_train_dtm)

# In[37]:

# convert sparse matrix to a dense matrix
features_matr = simple_train_dtm.toarray()
#print(type(features_matr))
#print(features_matr)


# examine the vocabulary and document-term matrix together
import pandas as pd
vocab = pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())

#insert class label and its value
vocab['class_label'] = train_set_label

#shuffle data and labels
"""import random
c = list(zip(vocab, labels))
random.shuffle(c)
vocab_random, labels_random = zip(*c)"""
#from sklearn.utils import shuffle
#vocab_random, labels_random = shuffle(vocab, labels, random_state=0)

#print(vocab)
#print(train_set_label)
print len(vocab)
print len(train_set_label)

#print(vocab)
#print(train_set_label)
#print(type(vocab))

#save dataset into file
#vocab.to_csv(r'trainset_feature.csv', header=True, index=True, sep=',')

