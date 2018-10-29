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
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle

# Function to load the dataset from a sqlite DB file
def load_data(database_filepath):
    '''
    Input: database_filepath    The full path and filename of the sqllite DB file containing the dataset
    
    Output: X                   The portion of the dataset to be used for training our model
            y                   The target columns that our model will be trained to predict
            category names      A list containing the names of these columns
            
    '''
    # Create the sqlite DB instance and read the contents into a DataFrame
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)

    # Create the 'y' DataFrame with only the category columns
    # Note that we drop the 'child_alone' column as it contains only 0 values
    y = df.drop(['id', 'message', 'original', 'genre', 'child_alone'], axis=1)
    
    # Create the 'X' DataFrame with only the 'message' column
    X = df['message']
    
    return X, y, list(y.columns)
    
# Function to tokenize text input    
def tokenize(text):
    '''
    Input:   text    Raw text
    Output:  A list containing the word tokens
    '''
    
    # We first remove punctuation characters and convert all text to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())

    # Get the word tokens
    raw_tokens = word_tokenize(text)
    
    # Remove (English) stop words
    clean_tokens = [t for t in raw_tokens if t not in stopwords.words("english")]
    
    # Lemmatize the tokens
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in clean_tokens]
    
    return lemmed

# Function that creates our machine learning pipeline, then sets up a GtridSearchCV object for selected parameters
def build_model():
    '''
    Input: None
    Output: A GridSearchCV object set up to iotimize our pipeline for selected parameters
    '''
    
    # Define the pipeline
    # Going with RandomForestClassifier after trying SVC
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])
    
    # SVC produced far worse prediction results than RandomForest
#    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
#                         ('tfidf', TfidfTransformer()),
#                        ('clf', MultiOutputClassifier(SVC(gamma='auto')))
#                    ])
    
    
    # Specify the grid search parameters
    # NOTE: Performing the grid search is very time consuming.
    # The final commit of this code has many of the parameters commented out as the search had to be run
    # with only a few parameters at a time to avoid the workspace timing out!
    
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

# Function to evaluate a model by predicting y values from the test set and creating a classification report
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Input:    model             The model to be used for prediction
              X_test            The test dataset used for the prediction
              Y_test            The values to be predicted
              category_names    The column names of the target dataset
              
    Output:   None
    '''
    
    # Run predict to create the prediction values
    y_pred = model.predict(X_test)
    
    # Print out the classification report containing the precision, recall and F1 scores for the model
    print(classification_report(Y_test, y_pred, target_names=category_names))

# Function to save a model to a pickle file
def save_model(model, model_filepath):
    '''
    Input:  model           The model to be saved
            model_filepath  THe full path and filename for the pickle file
            
    Output: None
    '''
    
    # Open a writeable file and store the pickled model
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