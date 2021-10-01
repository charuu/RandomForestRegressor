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

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt 
import seaborn as sns 
from math import sqrt

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def predict(model,X_test,y_test):
    y_pred=model.predict(X_test)
    print("Mean absolute error :", mean_absolute_error(y_test,y_pred))
  
    print("RMSE :", sqrt(mean_squared_error(y_test, y_pred)))
    
    plt.figure(figsize=(5, 7))

    ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
    sns.distplot(y_pred, hist=False, color="b", label="Fitted Values" , ax=ax)

    plt.title('Actual vs Fitted Values for Price')
    
    return plt

@click.command()
@click.argument('input_modelpath', type=click.Path(exists=True))
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_imagepath', type=click.Path())
def main(input_modelpath,input_filepath, output_imagepath):
    logger = logging.getLogger(__name__)
    logger.info('training model using processed data')
    
    loaded_model = pickle.load(open(os.path.join(input_modelpath,'model.sav'), 'rb'))
    X_test = pd.read_csv(os.path.join(input_filepath,"X_test.csv"))
    y_test = pd.read_csv(os.path.join(input_filepath,"y_test.csv"))
    X_test = X_test.drop(X_test.columns[0], axis=1)
    y_test = y_test.drop(y_test.columns[0], axis=1)
    
    plt = predict(loaded_model,X_test,y_test)
    plt.savefig(os.path.join(output_imagepath,'predict.png'))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
