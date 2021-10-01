# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os

from sklearn.impute import SimpleImputer

def data_normalisation(dataset):
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    scaled_features = scaler.transform(dataset)
    df_MinMax = pd.DataFrame(data=scaled_features, columns=dataset.columns)
    print(df_MinMax)
    interim_dataset = np.log1p(df_MinMax)
    
    return interim_dataset

def data_imputation(dataset):
    
    dataset[:] = dataset.replace([np.inf, -np.inf], np.nan)
    dataset_y = pd.DataFrame(dataset['y'],columns=['y'])
    dataset_x= dataset.drop(columns=['y'])
    
    limitPer = len(dataset_x) * 0.80
    dataset_x.dropna(thresh=limitPer, axis=1,inplace=True)
    
    data_imputation = pd.concat([dataset_x, dataset_y],axis=1)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imp.fit(data_imputation)   
    data_imputation[:] = imputer.transform(data_imputation)
    return data_imputation


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    dataset = pd.read_csv(os.path.join(input_filepath,"dataset_00_with_header.csv"))
    
    interim_dataset = data_imputation(dataset)
    interim_dataset.to_csv(os.path.join("data/interim","dataset_00_with_header.csv"))
    processed_dataset = data_normalisation(interim_dataset)
    
    
    processed_dataset.to_csv(os.path.join(output_filepath,"dataset_00_with_header.csv"))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
