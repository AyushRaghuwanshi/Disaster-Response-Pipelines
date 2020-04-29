# (Udacity)Disaster-Response-Pipelines

## Project Overview = 
  This project has been done during udaicity's Data Scienctist nanodegree.

  In the Project Workspace, we are having  a data set containing real messages that were sent during disaster events. We have to creat a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.
  
 ## Project Components = 
 There are three components in this projects = 
 ### 1. ETL Pipeline = 
 In a Python script, process_data.py, data cleaning pipeline is written that:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

### 2. ML Pipeline = 
In a Python script, train_classifier.py, machine learning pipeline is written that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

### 3 Flask Web App = 
Pre-made flask app ready to use we just need to change the filenames in this.

### Below are a few screenshots of the web app = 
