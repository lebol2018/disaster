# import libraries
import pandas as pd
from sqlalchemy import create_engine

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle

nltk.download(['punkt', 'wordnet', 'stopwords'])

# load data from database
engine = create_engine('sqlite:///messages.db')
df = pd.read_sql_table('Messages', engine)


df.head()

df = df[df.related < 2]
df.child_alone.value_counts()
y = df.drop(['id', 'message', 'original', 'genre', 'child_alone'], axis=1)
X = df['message']

X.head()

y.shape

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    raw_tokens = word_tokenize(text)
    clean_tokens = [t for t in raw_tokens if t not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in clean_tokens]
    return clean_tokens


pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])

X_train, X_test, y_train, y_test = train_test_split(X, y)
pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
yp = pd.DataFrame(y_pred)
print(classification_report(y_test, y_pred, target_names=list(y_test.columns)))

parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
#        'vect__max_df': (0.5, 0.75, 1.0),
#        'vect__max_features': (None, 5000, 10000),
#        'tfidf__use_idf': (True, False),
#        'clf__estimator__n_estimators': [50, 100, 200],
#        'clf__estimator__min_samples_split': [2, 3, 4]
}

cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)

for i in range(0,3):
    p = pipeline.steps[i]
    print(p[0], p[1].get_params().keys())
    print()

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
yp = pd.DataFrame(y_pred)
print(classification_report(y_test, y_pred, target_names=list(y_test.columns)))

file_name = 'model.pkl'
pickle.dump(cv.best_estimator_, open(file_name, "wb"))
