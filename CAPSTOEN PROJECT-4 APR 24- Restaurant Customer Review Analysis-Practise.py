import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#df = pd.read_csv(r'C:\Users\Neha\Downloads\NARESH IT DATA SCIENCE\APRIL\1- 13 APRIL\4 APR\Restaurant_Reviews.tsv')
# to resolve error use delimiter='\t' , quoting=3
df = pd.read_csv(r'C:\Users\Neha\Downloads\NARESH IT DATA SCIENCE\APRIL\1- 13 APRIL\4 APR\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv', delimiter='\t' , quoting=3)
df = pd.concat([df] * 10, ignore_index=True)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

wordnet = WordNetLemmatizer()
ps= PorterStemmer()
corpus=[]

#sentences = nltk.sent_tokenize(para[0])


# loop for removing stopwords:
for i in range(0,10000):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set (stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
tf= TfidfVectorizer()

x= tf.fit_transform(corpus).toarray()    
y= df.iloc[0:,1].values

# here we got x and y now we make ML model 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test= train_test_split(x,y,test_size=0.20,random_state=0)

# train dat ain Navie Bayes model
from sklearn.naive_bayes import BernoulliNB 
classifier= BernoulliNB(alpha=3.0 )
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test, y_pred)
print('Accuracy',ac)

bias = classifier.score(x_train, y_train)
print('bias',bias)

variance = classifier.score(x_test, y_test)
print('Variance',variance)

#================================================

def pred_unseen_dt(text):
    text = re.sub('[^a-zA-Z]',' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text]
    text = [' '.join(text)]
    X = tf.transform(text).toarray()
    y = classifier.predict(X)
    result = "Positive" if y[0] == 1 else "Negative"
    print(f'The customer gave {result} review')
    
pred_unseen_dt('the food was great')

pred_unseen_dt('the food was not great')

pred_unseen_dt('must try butter chicken they serve one of the best butter chicken')

pred_unseen_dt('Crust is not good.')

pred_unseen_dt('not liked')

pred_unseen_dt('food was not good')

pred_unseen_dt('loved it')

pred_unseen_dt('Now I am getting angry and I want my damn pho.')


# some sentences it gives wrong predictionsn but almost good but still need to tarin with more data

