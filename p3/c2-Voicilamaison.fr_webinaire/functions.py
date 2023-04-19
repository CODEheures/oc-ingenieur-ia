#!/usr/bin/env python
# coding: utf-8

import io
import shutil
import glob
import requests
from zipfile import ZipFile
from tempfile import mkdtemp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load(url: str) -> pd.DataFrame:
    # Request zip on AWS
    print(f'load data from {url}')
    response = requests.get(url)

    # Unzip file
    tempDir = mkdtemp()
    print(f'extract to temp dir: {tempDir}')

    with ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
        zip_ref.extractall(tempDir)

    csv = glob.glob(tempDir + '\*.csv')[0]

    # Read it in pandas
    print('Read csv by pandas')
    df = pd.read_csv(csv, delimiter='\t',
                     parse_dates=True,
                     low_memory=False)

    # Delete temp directory
    print('Delete temp Dir')
    shutil.rmtree(tempDir, ignore_errors=True)

    print('Dataset ready to explore under name: "df"!')
    return df


def infos(df: pd.DataFrame):
    memory_gb = np.round(df.memory_usage(deep=True).sum()/(1024**3),2)
    nb_lignes = df.shape[0]
    nb_columns = df.shape[1]
    print(f'A ce stade le dataset contient {nb_lignes} lignes et {nb_columns} colonnes. (conso mémoire {memory_gb}Gb)')


def extract_serie(dataFrame: pd.DataFrame, column: str) -> pd.Series:
    words = pd.Series(dtype=np.int32)
    for value in dataFrame.loc[:, column].dropna():
        splitted_values = value.split(',')
        for splitted_value in splitted_values:
            if splitted_value in words.keys():
                words[splitted_value] += 1
            else:
                words[splitted_value] = 0
    
    return words.sort_values(ascending=False)        


def scale(values: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values)
    pd.DataFrame(scaled_values).describe().round(2)
    return scaled_values


def acp(scaled_values, n_components = 2):
    pca = PCA(n_components=4)
    pca.fit(scaled_values)
    return pca.transform(scaled_values)


def acp_eboulis(pca: PCA):
    scree = (pca.explained_variance_ratio_*100).round(2)
    scree_cum = scree.cumsum()
    x = range(1, len(scree)+1)
    plt.bar(height=scree, x=x)
    plt.plot(x, scree_cum, '-o', c='r')

