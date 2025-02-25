import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
import nltk
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle
nltk.download(['punkt', 'wordnet'])

 New

def load_data(database_filepath):
    '''
    this function takes the filepath of database, split them into 
    input and target values.
    
    Arguments : 
        database filepath = from where data is to be load
    return : 
        X = list of input values
        Y = data frame of output values
        columns = labels of column 
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name='disasterresponse', con=engine)
    X = df.message.values
    Y = df.iloc[:,4:]
    columns = df.columns[4:]
    
    return X, Y, columns


def tokenize(text):
    '''
    this function takes the string and split it into
    tokkens followed by lemmatization and lower, strip operation on each
    tokken
    
    parameters :
        text = input string
        
    return :
        clean_tokens = list of tokens
    
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    
   '''
   this function creates model by using pipelins and 
   uses some parameter to select for best model parameters
   using gridesearchcv.
   uses randomforestclassifier.
   parameters:
    none
   return:
    cv : model
   
   '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'clf__estimator__criterion' : ['gini', 'entropy'],
    'clf__estimator__n_estimators' : [8,14,18]
    }
    
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters )
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    this functino evaluates the model on each column by using 
    classification_report.
    
    parameters:
        model = trained model
        X_test = input values for test set
        Y_test = target values for test set
        category_names = label names of output classes
    return:
        None
    '''
    y_pred_gs = model.predict(X_test)
    for i, col in enumerate(category_names):
        print('report for column {} = '.format(col))
        print(classification_report(Y_test.iloc[:,i], y_pred_gs[:,i], target_names=category_names))
    


def save_model(model, model_filepath):
    '''
    it takes the model and filepath to save the model as pickle object.
    parameters:
        model = trained model
        model_filepath = location where model is to be saved.
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
