import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from sklearn.datasets import load_files
from sklearn.metrics import plot_confusion_matrix
import pickle
from nltk.corpus import stopwords
import csv

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = WordNetLemmatizer()

reviews = [row for row in csv.reader(open('data_900.csv','r'))]
reviews = reviews[1:]
complaints = [row[0] for row in reviews]
topics = [row[1].lower() for row in reviews]

def process_text(complaint_list):
    documents = []
    for sen in range(0, len(complaint_list)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(complaint_list[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)

    return documents

documents = process_text(complaints)
#print(len(documents))

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
complaints = vectorizer.fit_transform(documents).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
complaints = tfidfconverter.fit_transform(documents).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(complaints, topics, test_size=0.2, random_state=0)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

#classifier = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
#classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
#print(accuracy_score(y_test, y_pred))

testdata = [row for row in csv.reader(open('test25.csv','r'))]

complaints_test = [row[0] for row in testdata]
#topics = [row[1].lower() for row in testdata]
for comp in complaints_test:
    print(classifier.predict(vectorizer.transform([comp])))

# Plot non-normalized confusion matrix
titles_options = [("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,

                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
