import pandas as pd
import numpy as np
from scipy.stats import stats

# Viene recuperato il dataset bank-data.csv
bd1 = pd.read_csv('bank-data.csv', sep=',')

# DATA CLEANING

# Eliminazione dei dati duplicati (secondo la key 'id')
bd1['id'].drop_duplicates()

# Eliminazione dei dati non veritieri: drop delle row aventi age <= 0 e income <= 0
bd1 = bd1[bd1['age'] > 0]
bd1 = bd1[bd1['income'] > 0]

#Rimozione dei valori outliers in age e income attraverso valutazione del z-score a 3 sigma di distanza
bd1[['age','income']] = bd1[['age','income']].astype('int64')
print(bd1['age'].describe())
print(bd1['income'].describe())
bd1[['age','income']] = bd1[['age','income']][(np.abs(stats.zscore(bd1[['age','income']])) < 3).all(axis=1)]
print(bd1['age'].describe())
print(bd1['income'].describe())

#Check e stampa dei missing values
def show_missing():
    missing = bd1.columns[bd1.isnull().any()].tolist()
    return missing
print('Missing values:\n',bd1[show_missing()].isnull().sum())


#Viene salvato il primo dataset da cui è stata rimossa la colonna 'id'
del bd1['id']
bd1.to_csv(r'C:\Users\lored\PycharmProjects\EserciziSIA_MarcoRuta\BankData\BankDataNotDiscretized.csv', index = False)


#Viene salvato il secondo dataset in cui è sono state discretizzate le features age e children
bd1['age'] = pd.cut( x = bd1['age'],
         bins =[0,10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
         labels = [10, 20, 30, 40, 50, 60, 70, 80, 90,100])

bd1['children'] = pd.Series(np.where(bd1.children.values > 0, 1, 0),bd1.index)

bd1.to_csv(r'C:\Users\lored\PycharmProjects\EserciziSIA_MarcoRuta\BankData\BankDataDiscretized.csv', index = False)

