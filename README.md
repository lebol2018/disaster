# Disaster Response Pipeline Project

In this project the assignment is to build an ETL and machine learning pipeline. Based on two datasets containing messages related to natural disasters and categories for classification, respectively, the task is to load and clean the data, save it to an SQLite database, then implement and train a machine learning pipeline that should be able to classify new messages based on the given categories.

### Dependencies:

* Python 3.6

Libraries used:
* pandas
* sqlalchemy
* nltk
* re
* sklearn
* pickle
* json
* plotly
* flask

### Files in this repository:

## /data
    * disaster_messages.csv, disaster_categories.csv
      The original datasets
    * process_data.py
      Contains code to load and clean the datasets, then store the resulting merged dataset in an SQLite database
    * DisasterResponse.db
      The SQLite database generated from process_data.py
      
## /models
    * train_classifier.py
      Contains code that loads the SQLite database, then instantiates, trains and evaluates a machine learning model. This model is then         optimized using grid search, and the resulting model is saved tob a Pickle file.
    * classfier.pkl
      The pickled machine learning model saved from train_classifier.py
      
## /app
    * run.py
      A Flask web application that 
      1. Renders plots of a couple of chosen characteristics in the dataset
      2. Upon receiving a text query, responds with a classification of that text using the trained machine learning model
     
     
### Usage
      * To load and clean the data
       python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
       
       * To train the machine learning model
       python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
       
       * To run the Flask web app
       cd app
       python run.py
       
       Go to http://0.0.0.0:3001/
       
       
