#!/usr/bin/env python
# coding: utf-8

# ### Fake News Classifier
# Dataset:  https://www.kaggle.com/c/fake-news/data#

# In[34]:


import pandas as pd
import nltk
import pickle
import re


# In[19]:


df=pd.read_csv('train.csv')


# In[20]:


df.head()


# In[21]:


## Get the Independent Features

X=df.drop('label',axis=1)


# In[22]:


X.head()


# In[23]:


## Get the Dependent features
y=df['label']


# In[24]:


y.head()


# In[25]:


df.shape


# In[26]:


from sklearn.feature_extraction.text import CountVectorizer
# In[27]:


df=df.dropna()


# In[28]:


df.head(10)


# In[29]:


messages=df.copy()


# In[30]:


messages.reset_index(inplace=True)


# In[31]:


messages.head(10)


# In[32]:


messages['title'][6]


# In[35]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
lm = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[36]:


corpus[3]


# In[37]:


## Applying Countvectorizer
# Creating the Bag of Words model
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()


# In[38]:

# Creating a pickle file for the CountVectorizer
pickle.dump(cv, open('fakenewscv.pkl', 'wb'))
X.shape


# In[39]:


y=messages['label']


# In[40]:


## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[41]:


cv.get_feature_names()[:20]


# In[42]:


cv.get_params()


# In[43]:


count_df = pd.DataFrame(X_train, columns=cv.get_feature_names())


# In[44]:


count_df.head()


# In[45]:


import matplotlib.pyplot as plt


# In[46]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ### MultinomialNB Algorithm

# In[47]:



from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


# In[48]:


from sklearn import metrics
import numpy as np
import itertools


# In[49]:


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[50]:


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
score


# In[51]:

# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'fakenewsclassifier.pkl'
pickle.dump(classifier, open(filename, 'wb'))
y_train.shape


# 
# 

# In[ ]:





# In[ ]:





# ### Multinomial Classifier with Hyperparameter

# In[64]:


classifier=MultinomialNB(alpha=0.1)


# In[65]:


previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))


# In[66]:


## Get Features names
feature_names = cv.get_feature_names()


# In[67]:


classifier.coef_[0]


# In[68]:


### Most real
sorted(zip(classifier.coef_[0], feature_names), reverse=True)[:20]


# In[69]:


### Most fake
sorted(zip(classifier.coef_[0], feature_names))[:5000]


# In[ ]:




