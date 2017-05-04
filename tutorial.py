import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_iris
import os
import email.parser
import re
from email.parser import Parser
from email.Utils import parseaddr
from email.Header import decode_header

iris = load_iris()
# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target
# ## Part 3: Reading a text-based dataset into pandas

# In[38]:

atom_rfc2822=r"[a-zA-Z0-9_!#\$\%&'*+/=?\^`{}~|\-]+"
atom_posfix_restricted=r"[a-zA-Z0-9_#\$&'*+/=?\^`{}~|\-]+" # without '!' and '%'
atom=atom_rfc2822
dot_atom=atom  +  r"(?:\."  +  atom  +  ")*"
quoted=r'"(?:\\[^\r\n]|[^\\"])*"'
local="(?:"  +  dot_atom  +  "|"  +  quoted  +  ")"
domain_lit=r"\[(?:\\\S|[\x21-\x5a\x5e-\x7e])*\]"
domain="(?:"  +  dot_atom  +  "|"  +  domain_lit  +  ")"
addr_spec=local  +  "\@"  +  domain

email_address_re=re.compile('^'+addr_spec+'$')

def getmailaddresses(msg, name):
    """retrieve From:, To: and Cc: addresses"""
    addrs=email.utils.getaddresses(msg.get_all(name, []))
    for i, (name, addr) in enumerate(addrs):
        if not name and addr:
            # only one string! Is it the address or is it the name ?
            # use the same for both and see later
            name=addr
            
        try:
            # address must be ascii only
            addr=addr.encode('ascii')
        except UnicodeError:
            addr=''
        else:
            # address must match adress regex
            if not email_address_re.match(addr):
                addr=''
        addrs[i]=(getmailheader(name), addr)
    return addrs

def getmailheader(header_text, default="ascii"):
    """Decode header_text if needed"""
    try:
        headers=decode_header(header_text)
    except email.Errors.HeaderParseError:
        # This already append in email.base64mime.decode()
        # instead return a sanitized ascii string 
        return header_text.encode('ascii', 'replace').decode('ascii')
    else:
        for i, (text, charset) in enumerate(headers):
            try:
                headers[i]=unicode(text, charset or default, errors='replace')
            except LookupError:
                # if the charset is unknown, force default 
                headers[i]=unicode(text, default, errors='replace')
        return u"".join(headers)

# read file into pandas using a relative path
#path = 'data/sms.tsv'
#path = 'data/0006.2003-12-18.GP.spam.txt'
parser = Parser()
#rootdir = '/root/Desktop/Machine_Learning/Project-SpamDetection/'
rootdir = 'ham/beck-s/wilson__shona'
listtexts = []
labels = []
for subdirs,dir,files in os.walk(rootdir):
    for file in files:
        path =  os.path.join(subdirs, file)
        if '.idea' in path:
            continue
        elif 'py' in path:
            continue
        else:
            f = open(path,'r').read()
            msg = email.message_from_string(f)
            subject=getmailheader(msg.get('Subject', ''))
            print(subject)
            from_=getmailaddresses(msg, 'from')
            from_=('', '') if not from_ else from_[0]
            print(from_)
            if msg.is_multipart():
                for payload in msg.get_payload():
                    Text = str(payload.get_payload())
            else:
                Text = str(msg.get_payload())

            cleanbr = re.compile('<br>|<BR>')
            cleanr = re.compile('<.*?>')
            #cleannline = re.compile('\n')
            Text = re.sub('\s+', ' ', Text)
            #Text = Text.translate("  ", '\t\n ')
            Text = re.sub(cleanbr, ' ', Text)
            Text = re.sub(cleanr, '', Text)
        
            '''email = f.read()
            em = email.splitlines()

            Text = ""
            flag = 0

            for e in em:
                if 'From:' in e and e[0] == 'F' and e[4] == ':':
                    print(e)
                    listTo.append(e)
                elif 'To:' in e and e[0] == 'T' and e[2] == ':':
                    print(e)
                    listFrom.append(e)

                if 'X-FileName:' in e:
                    flag = 1
                    continue
                if flag == 1:
                    Text = Text + e'''
        listtexts.append(Text)
        if 'BG' in path or 'GP' in path or 'SH' in path:
            labels.append('Spam')
        else:
            labels.append('Not Spam')
print listtexts
#print labels
#sms = pd.read_table(path, header=None, names=['label', 'message'])


# In[ ]:

# alternative: read file into pandas from a URL
# url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
# sms = pd.read_table(url, header=None, names=['label', 'message'])


# In[39]:

# examine the shape
#sms.shape
#email


# In[41]:


#print ("test")
#print (em)


# In[33]:

#type(em)


# In[34]:


vect = CountVectorizer(stop_words='english')
# learn the 'vocabulary' of the training data (occurs in-place)
vect.fit(listtexts)


# In[35]:

# examine the fitted vocabulary
vect.get_feature_names()


# In[36]:

# transform training data into a 'document-term matrix'
simple_train_dtm = vect.transform(listtexts)
simple_train_dtm


# In[37]:

# convert sparse matrix to a dense matrix
simple_train_dtm.toarray()


# In[ ]:

# examine the first 10 rows
#sms.head(10)


# In[ ]:

# examine the class distribution
#sms.label.value_counts()


# In[ ]:

# convert label to a numerical variable
#sms['label_num'] = sms.label.map({'ham':0, 'spam':1})


# In[ ]:

# check that the conversion worked
#sms.head(10)


# In[ ]:

# how to define X and y (from the iris data) for use with a MODEL
#X = iris.data
#y = iris.target
#print(X.shape)
#print(y.shape)


# In[ ]:

# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
#X = sms.message
#y = sms.label_num
#print(X.shape)
#print(y.shape)


# In[ ]:

# split X and y into training and testing sets
'''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# ## Part 4: Vectorizing our dataset

# In[ ]:

# instantiate the vectorizer
vect = CountVectorizer()


# In[ ]:

# learn training data vocabulary, then use it to create a document-term matrix
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)


# In[ ]:

# equivalently: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)


# In[ ]:

# examine the document-term matrix
X_train_dtm


# In[ ]:

# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm


# ## Part 5: Building and evaluating a model
# 
# We will use [multinomial Naive Bayes](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html):
# 
# > The multinomial Naive Bayes classifier is suitable for classification with **discrete features** (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.

# In[ ]:

# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[ ]:

# train the model using X_train_dtm (timing it with an IPython "magic command")
get_ipython().magic(u'time nb.fit(X_train_dtm, y_train)')


# In[ ]:

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)


# In[ ]:

# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# In[ ]:

# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# In[ ]:

# print message text for the false positives (ham incorrectly classified as spam)


# In[ ]:

# print message text for the false negatives (spam incorrectly classified as ham)


# In[ ]:

# example false negative
X_test[3132]


# In[ ]:

# calculate predicted probabilities for X_test_dtm (poorly calibrated)
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob


# In[ ]:

# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)


# ## Part 6: Comparing models
# 
# We will compare multinomial Naive Bayes with [logistic regression](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression):
# 
# > Logistic regression, despite its name, is a **linear model for classification** rather than regression. Logistic regression is also known in the literature as logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

# In[ ]:

# import and instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[ ]:

# train the model using X_train_dtm
get_ipython().magic(u'time logreg.fit(X_train_dtm, y_train)')


# In[ ]:

# make class predictions for X_test_dtm
y_pred_class = logreg.predict(X_test_dtm)


# In[ ]:

# calculate predicted probabilities for X_test_dtm (well calibrated)
y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]
y_pred_prob


# In[ ]:

# calculate accuracy
metrics.accuracy_score(y_test, y_pred_class)


# In[ ]:

# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)


# ## Part 7: Examining a model for further insight
# 
# We will examine the our **trained Naive Bayes model** to calculate the approximate **"spamminess" of each token**.

# In[ ]:

# store the vocabulary of X_train
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)


# In[ ]:

# examine the first 50 tokens
print(X_train_tokens[0:50])


# In[ ]:

# examine the last 50 tokens
print(X_train_tokens[-50:])


# In[ ]:

# Naive Bayes counts the number of times each token appears in each class
nb.feature_count_


# In[ ]:

# rows represent classes, columns represent tokens
nb.feature_count_.shape


# In[ ]:

# number of times each token appears across all HAM messages
ham_token_count = nb.feature_count_[0, :]
ham_token_count


# In[ ]:

# number of times each token appears across all SPAM messages
spam_token_count = nb.feature_count_[1, :]
spam_token_count


# In[ ]:

# create a DataFrame of tokens with their separate ham and spam counts
tokens = pd.DataFrame({'token':X_train_tokens, 'ham':ham_token_count, 'spam':spam_token_count}).set_index('token')
tokens.head()


# In[ ]:

# examine 5 random DataFrame rows
tokens.sample(5, random_state=6)


# In[ ]:

# Naive Bayes counts the number of observations in each class
nb.class_count_


# Before we can calculate the "spamminess" of each token, we need to avoid **dividing by zero** and account for the **class imbalance**.

# In[ ]:

# add 1 to ham and spam counts to avoid dividing by 0
tokens['ham'] = tokens.ham + 1
tokens['spam'] = tokens.spam + 1
tokens.sample(5, random_state=6)


# In[ ]:

# convert the ham and spam counts into frequencies
tokens['ham'] = tokens.ham / nb.class_count_[0]
tokens['spam'] = tokens.spam / nb.class_count_[1]
tokens.sample(5, random_state=6)


# In[ ]:

# calculate the ratio of spam-to-ham for each token
tokens['spam_ratio'] = tokens.spam / tokens.ham
tokens.sample(5, random_state=6)


# In[ ]:

# examine the DataFrame sorted by spam_ratio
# note: use sort() instead of sort_values() for pandas 0.16.2 and earlier
tokens.sort_values('spam_ratio', ascending=False)


# In[ ]:

# look up the spam_ratio for a given token
tokens.loc['dating', 'spam_ratio']


# ## Part 8: Practicing this workflow on another dataset
# 
# Please open the **`exercise.ipynb`** notebook (or the **`exercise.py`** script).

# ## Part 9: Tuning the vectorizer (discussion)
# 
# Thus far, we have been using the default parameters of [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html):

# In[ ]:

# show default parameters for CountVectorizer
vect


# However, the vectorizer is worth tuning, just like a model is worth tuning! Here are a few parameters that you might want to tune:
# 
# - **stop_words:** string {'english'}, list, or None (default)
#     - If 'english', a built-in stop word list for English is used.
#     - If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
#     - If None, no stop words will be used.

# In[6]:

# remove English stop words
vect = CountVectorizer(stop_words='english')


# - **ngram_range:** tuple (min_n, max_n), default=(1, 1)
#     - The lower and upper boundary of the range of n-values for different n-grams to be extracted.
#     - All values of n such that min_n <= n <= max_n will be used.

# In[7]:

# include 1-grams and 2-grams
vect = CountVectorizer(ngram_range=(1, 2))


# - **max_df:** float in range [0.0, 1.0] or int, default=1.0
#     - When building the vocabulary, ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
#     - If float, the parameter represents a proportion of documents.
#     - If integer, the parameter represents an absolute count.

# In[ ]:

# ignore terms that appear in more than 50% of the documents
vect = CountVectorizer(max_df=0.5)


# - **min_df:** float in range [0.0, 1.0] or int, default=1
#     - When building the vocabulary, ignore terms that have a document frequency strictly lower than the given threshold. (This value is also called "cut-off" in the literature.)
#     - If float, the parameter represents a proportion of documents.
#     - If integer, the parameter represents an absolute count.

# In[8]:

# only keep terms that appear in at least 2 documents
vect = CountVectorizer(min_df=2)


# **Guidelines for tuning CountVectorizer:**
# 
# - Use your knowledge of the **problem** and the **text**, and your understanding of the **tuning parameters**, to help you decide what parameters to tune and how to tune them.
# - **Experiment**, and let the data tell you the best approach!
'''