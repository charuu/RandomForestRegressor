# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
import pickle
import seaborn as sns 
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def build_features(X_train, y_train):
    logger = logging.getLogger(__name__)
    logger.info('Create random forest regressor model')
    
    rfc = RandomForestRegressor(n_estimators = 50)
    
    logger.info('Select minimal features')
    
    features = SelectFromModel(rfc)
    features.fit(X_train, y_train)
    
    selected_feat= X_train.columns[features.get_support()]
    print('Selected values: ',selected_feat.values)
    
    X_train = X_train[selected_feat.values]
    
    return rfc, selected_feat.values

def split_dataset(dataset):
    dataset_x= dataset.drop(columns=['y'])
    dataset_y = dataset['y']
    return train_test_split(dataset_x, dataset_y, test_size=0.3) # 70% training and 30% test
    
def train(dataset, input_filepath):
    logger = logging.getLogger(__name__)
    X_train, X_test, y_train, y_test = split_dataset(dataset) 
    
    rfc, features = build_features(X_train, y_train)
    
    logger.info('Fit training data to random forest regressor model')
    
    rfc.fit(X_train[features], y_train)

    return rfc, X_test[features], y_test

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_modelpath', type=click.Path())
def main(input_filepath, output_modelpath):
    logger = logging.getLogger(__name__)
    logger.info('training model using processed data')
    dataset = pd.read_csv(os.path.join(input_filepath,"dataset_00_with_header.csv"))
    
    model, X_test, y_test  = train(dataset, input_filepath)
    
    X_test.to_csv(os.path.join(input_filepath,"X_test.csv"))
    y_test.to_csv(os.path.join(input_filepath,"y_test.csv"))
    
    logger.info('Writing model')
    pickle.dump(model, open(os.path.join(output_modelpath,"model.sav"), 'wb'))
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
