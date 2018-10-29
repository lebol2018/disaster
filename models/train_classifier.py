import sys
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

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    df = df[df.related < 2]
    y = df.drop(['id', 'message', 'original', 'genre', 'child_alone'], axis=1)
    X = df['message']
    
    return X, y, list(y.columns)
    
    
def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    raw_tokens = word_tokenize(text)
    clean_tokens = [t for t in raw_tokens if t not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in clean_tokens]
    
    return lemmed

def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])
    
    parameters = {
    #        'vect__ngram_range': ((1, 1), (1, 2)),
    #        'vect__max_df': (0.5, 0.75, 1.0),
    #        'vect__max_features': (None, 5000, 10000),
             'tfidf__use_idf': (True, False),
    #        'clf__estimator__n_estimators': [50, 100, 200],
    #        'clf__estimator__min_samples_split': [2, 3, 4]
    }    
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    yp = pd.DataFrame(y_pred)
    print(classification_report(Y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        nltk.download(['punkt', 'wordnet', 'stopwords'])

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()