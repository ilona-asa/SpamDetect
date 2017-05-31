print(__doc__)

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


h = .02  # step size in the mesh

names = ["k-NN", "Linear SVM", "Neural Net", "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(),
    LinearSVC(),
    MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto', 
       beta_1=0.9, beta_2=0.999, early_stopping=False, 
       epsilon=1e-08, hidden_layer_sizes=(5, 2), 
       learning_rate='constant', learning_rate_init=0.001, 
       max_iter=200, momentum=0.9, nesterovs_momentum=True, 
       power_t=0.5, random_state=1, shuffle=True, solver='lbfgs', 
       tol=0.0001, validation_fraction=0.1, verbose=False, 
       warm_start=False),
    MultinomialNB()]

"""X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]
"""
########## start prepare dataset ###############
rootdir = 'preprocessed_data/enron1/'
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
            labels.append(0)
            #labels.append('Spam')
        else:
            labels.append(1)
            #labels.append('Ham')


from sklearn.utils import shuffle
listtexts_random, labels_random = shuffle(listtexts, labels, random_state=0)

def splitData(textdata, label, trainSize):
    #split = int(trainSize*len(textdata))
    split = 100
    n = 50
    train_set_text = textdata[:split]
    train_set_label = label[:split]
    test_set_text = textdata[split:n]
    test_set_label = label[split:n]
    return train_set_text, train_set_label, test_set_text, test_set_label

#train_set_text, train_set_label, test_set_text, test_set_label = splitData(listtexts_random, labels_random, 0.7)

##Vectorized the data
vect = CountVectorizer(stop_words='english')
# learn the 'vocabulary' of the training data (occurs in-place)
vect.fit(listtexts_random)

# examine the fitted vocabulary
features_name = vect.get_feature_names()
#print(type(features))
print(len(features_name))

# transform training data into a 'document-term matrix'
simple_train_dtm = vect.transform(listtexts_random)

# convert sparse matrix to a dense matrix
features_matr = simple_train_dtm.toarray()

# examine the vocabulary and document-term matrix together
features = pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
#print(len(features))
feature_nparr = features.as_matrix()

from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.decomposition import TruncatedSVD
pca = PCA(n_components=2, svd_solver='randomized')
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
svd = TruncatedSVD(n_components=2, algorithm='randomized')
#X = pca.fit_transform(feature_nparr)
#X = svd.fit_transform(feature_nparr)
X = tsne.fit_transform(feature_nparr)
#x, y = X_trans[:,0], X_trans[:,1]


##split data
train_set_text, train_set_label, test_set_text, test_set_label = splitData(X, labels_random, 0.9)

########## end prepare dataset #################


figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
#for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
#    X, y = ds
#    X = StandardScaler().fit_transform(X)
#    X_train, X_test, y_train, y_test = \
#        train_test_split(X, y, test_size=.4, random_state=42)


x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1, len(classifiers) + 1, i)
#    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
#    if ds_cnt == 0:
#        ax.set_title("Input data")
# Plot the training points
ax.scatter(train_set_text[:, 0], train_set_text[:, 1], c=train_set_label, cmap=cm_bright)
# and testing points
ax.scatter(test_set_text[:, 0], test_set_text[:, 1], c=test_set_label, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 1

# iterate over classifiers
for name, clf in zip(names, classifiers):
    #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    ax = plt.subplot(1, len(classifiers) + 1, i)
    clf.fit(train_set_text, train_set_label)
    score = clf.score(test_set_text, test_set_label)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(train_set_text[:, 0], train_set_text[:, 1], c=train_set_label, cmap=cm_bright)
    # and testing points
    ax.scatter(test_set_text[:, 0], test_set_text[:, 1], c=test_set_label, cmap=cm_bright, alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    #if ds_cnt == 0:
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
    i += 1

plt.tight_layout()
plt.show()
