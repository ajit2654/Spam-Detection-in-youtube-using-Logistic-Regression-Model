import re

import matplotlib.pyplot as plt
import numpy as num
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier


def drop_columns(columns,data):
    data.drop(columns, axis=1, inplace=True )

def process_contents(content):
    return " ".join(re.findall("[A-Za-z]+",content.lower()))




if __name__=="__main__":
    training_dataset = []
    dataset = ['//home//ajit//Documents//YouTube-Spam-Collection-v1//Youtube01-Psy.csv','//home//ajit//Documents//YouTube-Spam-Collection-v1//Youtube02-KatyPerry.csv','//home//ajit//Documents//YouTube-Spam-Collection-v1//Youtube03-LMFAO.csv','//home//ajit//Documents//YouTube-Spam-Collection-v1//Youtube04-Eminem.csv','//home//ajit//Documents//YouTube-Spam-Collection-v1//Youtube05-Shakira.csv']

    for files in dataset:
        read_data = pd.read_csv(files)
        training_dataset.append(read_data)

    training_dataset = pd.concat(training_dataset)
    #training_dataset.info()

    #print(training_dataset)

    #print(training_dataset['CLASS'].value_counts())
    drop_columns(['COMMENT_ID','AUTHOR','DATE'],training_dataset)
    print(training_dataset.info())
    training_dataset['Processed_Content'] = training_dataset['CONTENT'].apply(process_contents)
    #print(training_dataset.head(200))
    drop_columns(['CONTENT'],training_dataset)
    training_dataset.info()
    #print(training_dataset.head(200))


    x_train, x_test, y_train, y_test = train_test_split(training_dataset['Processed_Content'],training_dataset['CLASS'],test_size=0.2,random_state=57)

    count_vect = CountVectorizer(stop_words='english')
    x_train_counts = count_vect.fit_transform(x_train)
    #x_train_counts.shape
    tranformer = TfidfTransformer()
    x_train_tfidf = tranformer.fit_transform(x_train_counts)
    #x_train_tfidf.shape
    x_test_counts = count_vect.transform(x_test)

    x_test_tfidf = tranformer.transform(x_test_counts)
    ## Random forest classifier model
     print('Spam Detection using Random Forest Classifier Model')
    model = RandomForestClassifier()
    model.fit(x_train_tfidf,y_train)
    predictions = model.predict(x_test_tfidf)

    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))

    ## Spam Detection using Logistic Regression Model
    print('Spam Detection using Logistic Regression Model')
    model = LogisticRegression()
    model.fit(x_train_tfidf,y_train)
    predictions = model.predict(x_test_tfidf)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    
