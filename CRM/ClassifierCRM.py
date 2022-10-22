import pandas as pd
import openpyxl
import numpy as np
import matplotlib.pylab as plt
import sklearn
from scipy.stats import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

# DATA CLEANING
print('####Inizio operazioni di data cleaning...####')

# Lettura del dataset e salvataggio nel dataframe crm1
crm1 = pd.read_excel('CRM data.xlsx')

# Sposto l'attributo target 'Y' come ultima colonna
column_names = ["client_code", "installment", "first_amount_spent", "number_of_products", "dim_small", "dim_medium",
                "dim_big", "north", "south_and_islands", "center", "age51_89", "age36_50", "age15_35", "sex", "Y"]
crm1 = crm1.reindex(columns=column_names)


# Funzione di check dei missing values
def show_missing():
    missing = crm1.columns[crm1.isnull().any()].tolist()
    return missing


# Stampa dei missing values
print('Lista dei valori Null:\n', crm1[show_missing()].isnull().sum())

# Drop di 19 righe contenenti valori NULL
crm1 = crm1[crm1.south_and_islands.notnull()]
crm1 = crm1[crm1.dim_small.notnull()]
crm1 = crm1[crm1.north.notnull()]
print('Lista dei valori Null:\n', crm1[show_missing()].isnull().sum())

# Rimozione delle righe con client_code duplicato
crm1.drop_duplicates(subset='client_code')

# Eliminazione di una riga con client_code in formato errato
crm1 = crm1[crm1.client_code.map(len) < 10]

# Eliminazione di valori non numerici errati in first amount spent
crm1 = crm1[crm1['first_amount_spent'].apply(lambda x: str(x).isdigit())]

# Modifica dei valori errati in number of products e first_amount_spent, il valore 0 non ha senso
crm1.loc[crm1.number_of_products == 0, 'number_of_products'] = int(crm1['number_of_products'].mean())
crm1.loc[crm1.first_amount_spent == 0, 'first_amount_spent'] = int(crm1['first_amount_spent'].mean())

# Rimozione dei valori outliers nei campi first_amount_spent e number_of_products attraverso
# z-score a 3 * dev.std di distanza massima
z_scores = stats.zscore(np.array(crm1[['first_amount_spent', 'number_of_products']], dtype=np.float64))
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
crm1 = crm1[filtered_entries]

# One-hot encoding di client_code, converto l'intero dataframe in numeric
encoding_list = ['client_code']
crm1[encoding_list] = crm1[encoding_list].apply(LabelEncoder().fit_transform)
crm1 = crm1.apply(pd.to_numeric)
print(crm1.dtypes)

# Viene salvato il nuovo dataset cleaned
crm1.to_excel(r'C:\Users\lored\PycharmProjects\EserciziSIA_MarcoRuta\CRM\CRM data cleaned.xlsx', index=False)

print('####Data cleaning completato!####')

# Viene recuperato il nuovo dataset cleaned
crm2 = pd.read_excel('CRM data cleaned.xlsx')

# Prima analisi sulla quantità di prodotti acquistati attraverso istogrammi
crm2.hist(bins=10, figsize=(20, 15), color='#E14906')
plt.tight_layout(pad=4)
plt.show()

# Calcolo della correlazione di tutte le features con il target 'Y'
corr = crm2.corr()["Y"]
corr[np.argsort(corr, axis=0)[::-1]]

# Plot della correlazione di tutte le features con la feature 'Y'
num_feat = crm2.columns[crm2.dtypes != object]
num_feat_Y = num_feat[0:15]
num_feat_noY = num_feat[0:14]
labels = []
values = []
for col in num_feat_noY:
    labels.append(col)
    values.append(np.corrcoef(crm2[col].values, crm2.Y.values)[0, 1])

ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12, 8))
rects = ax.barh(ind, np.array(values), color='green')
ax.set_yticks(ind + (width/ 2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Coefficente di relazione")
ax.set_title("Correlazione con la features Y");
plt.show()

# Heatmap della correlazione tra tutti gli attributi
corrMatrix = crm2[num_feat].corr()
sns.set(font_scale=1.10)
plt.figure(figsize=(10, 10))
sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True, annot=True, cmap='viridis', linecolor="white")

plt.title('Correlation between features');
plt.tight_layout(pad=4)
plt.show()

# Preparazione del dataset per i modelli
x_data = crm2.drop('Y', axis=1)
y_data = crm2['Y']

# splitting di training  e testing set
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

# Preparazione e dichiarazione dei classificatori utilizzati
names = ['Nearest Neighbors',
         'Decision Tree',
         'Random Forest',
         'Naive Bayes',
         'Logistic Regression',
         'Gradient Boosting',
         'Gaussian Process']

classifiers = [
    KNeighborsClassifier(10),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB(),
    LogisticRegression(solver="lbfgs", max_iter=500),
    GradientBoostingClassifier(n_estimators=1000),
    GaussianProcessClassifier()
]

# Viene addestrato ogni classificatore sul training set e poi viene calcolata l'accuracy sul testing set
for name, clf in zip(names, classifiers):
    print('\n', name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print('Accuracy: %0.2f' % accuracy_score(y_test, y_pred))

# Cross validation e Matrice di confusione del modello decision Tree
print('Cross validation e Matrice di confusione del modello Decision Tree\n')
dt = DecisionTreeClassifier(max_depth=5)
scores = cross_val_score(dt, X_train, y_train, cv=10, scoring="accuracy")
print('Scores: ', scores)
print('Media: ', scores.mean())
print('Deviazione standard: ', scores.std(), '\n')

# Matrice di confusione del modello DecisionTree
predictions = cross_val_predict(DecisionTreeClassifier(), X_train, y_train, cv=3)
matrice = confusion_matrix(y_train, predictions)
print('Matrice di confusione: \n', matrice)

# Cross validation e Matrice di confusione del modello Gaussian Process
print('Cross validation e Matrice di confusione del modello GaussianProcess\n')
gp = GaussianProcessClassifier()
scores = cross_val_score(gp, X_train, y_train, cv=10, scoring="accuracy")
print('Scores: ', scores)
print('Media: ', scores.mean())
print('Deviazione standard: ', scores.std(), '\n')

# Matrice di confusione del modello Gaussian Process
predictions = cross_val_predict(gp, X_train, y_train, cv=3)
matrice = confusion_matrix(y_train, predictions)
print('Matrice di confusione: \n', matrice)
