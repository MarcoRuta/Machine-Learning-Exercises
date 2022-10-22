import pandas as pd
import matplotlib.pylab as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# DATA CLEANING
print('####Inizio operazioni di data cleaning...####')

# Lettura del dataset e salvataggio nel dataframe nd1
nd1 = pd.read_csv('nursery data.csv', sep=',')

# Funzione di check dei missing values
def show_missing():
    missing = nd1.columns[nd1.isnull().any()].tolist()
    return missing

# Stampa dei missing values
print('Lista dei valori Null:\n',nd1[show_missing()].isnull().sum())

# Eliminazione dei caratteri incorretti (mistyping) presenti nel campo class
nd1['class'] = nd1['class'].str.replace(r';', '')

print('####Fine operazioni di data cleaning!####')

# Prima analisi dei dati attraverso istogrammi
figure, axis = plt.subplots(3, 3)

nd1['parents'].hist(bins=10, figsize=(20,15), color='#E14906', ax = axis[0,0])
figure.tight_layout(pad=4)
axis[0, 0].set_title("parents")

nd1['has_nurs'].hist(bins=10, figsize=(20,15), color='#E14906', ax = axis[0,1])
figure.tight_layout(pad=4)
axis[0, 1].set_title("parents")

nd1['form'].hist(bins=10, figsize=(20,15), color='#E14906', ax = axis[0,2])
figure.tight_layout(pad=4)
axis[0, 2].set_title("form")

nd1['children'].hist(bins=10, figsize=(20,15), color='#E14906', ax = axis[1,0])
figure.tight_layout(pad=4)
axis[1, 0].set_title("children")

nd1['housing'].hist(bins=10, figsize=(20,15), color='#E14906', ax = axis[1,1])
figure.tight_layout(pad=4)
axis[1, 1].set_title("housing")

nd1['finance'].hist(bins=10, figsize=(20,15), color='#E14906', ax = axis[1,2])
figure.tight_layout(pad=4)
axis[1, 2].set_title("finance")

nd1['social'].hist(bins=10, figsize=(20,15), color='#E14906', ax = axis[2,0])
figure.tight_layout(pad=4)
axis[2, 0].set_title("social")

nd1['health'].hist(bins=10, figsize=(20,15), color='#E14906', ax = axis[2,1])
figure.tight_layout(pad=4)
axis[2, 1].set_title("health")

nd1['class'].hist(bins=10, figsize=(20,15), color='#E14906', ax = axis[2,2])
figure.tight_layout(pad=4)
axis[2, 2].set_title("class")

plt.show()

# One-hot encoding di tutti i dati categoriali attraverso labelencoder
encoding_list = ['parents', 'has_nurs', 'form', 'housing', 'finance', 'social', 'health', 'children', 'class']

for col in encoding_list:
    le = LabelEncoder()
    col_values_unique = list(nd1[col].unique())
    le_fitted = le.fit(col_values_unique)

    col_values = list(nd1[col].values)
    le.classes_
    col_values_transformed = le.transform(col_values)
    nd1[col] = col_values_transformed

# Viene salvato il nuovo dataset cleaned
nd1.to_csv(r'C:\Users\lored\PycharmProjects\EserciziSIA_MarcoRuta\NurseryData\NurseryDataEncoded.csv', index=False)

# Viene recuperato il nuovo dataset cleaned
nd2 = pd.read_csv('NurseryDataEncoded.csv')


# Preparazione del dataset per i modelli
x_data = nd2.drop('class', axis = 1)
y_data = nd2['class']

# splitting di training  e testing set
X_train, X_test, y_train, y_test = train_test_split( x_data, y_data, test_size=1 / 3, random_state=0 )

# Preparazione e dichiarazione dei classificatori utilizzati
names = ['Nearest Neighbors',
         'Decision Tree',
         'Random Forest',
         'Naive Bayes',
         'Logistic Regression',
         'Gradient Boosting']

classifiers = [
    KNeighborsClassifier(10),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier( max_depth=5, n_estimators=10, max_features=1),
    GaussianNB(),
    LogisticRegression(solver="lbfgs", max_iter=50000),
    GradientBoostingClassifier(n_estimators=100),
]

# Viene addestrato ogni classificatore clf sul training set e poi viene calcolata l'accuracy sul testing set
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('\n',name,accuracy_score(y_test, y_pred))
    print('Accuracy: %0.2f\n' % accuracy_score(y_test, y_pred))

# Cross validitation del modello cdf
    print('Cross validation e Matrice di confusione del modello ',name,'\n')
    scores = cross_val_score(clf, X_train, y_train, cv=2, scoring="accuracy")
    print('Scores: ', scores)
    print('Media: ', scores.mean())
    print('Deviazione standard: ', scores.std(), '\n')

    # Matrice di confusione del modello clf
    predictions = cross_val_predict(clf, X_train, y_train, cv=2)
    matrice = confusion_matrix(y_train, predictions)
    print('Matrice di confusione: \n', matrice)
