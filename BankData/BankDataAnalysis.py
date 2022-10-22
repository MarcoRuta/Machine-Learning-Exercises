import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import wittgenstein as lw
from matplotlib import gridspec
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Import del dataset con le features age e children non discretizzate
bdNotDisc = pd.read_csv('BankDataNotDiscretized.csv', sep=',')

# Import del dataset con le features age e children discretizzate
bdDisc = pd.read_csv('BankDataDiscretized.csv', sep=',')

# Conversione one hot encoding dei dati categoriali sex, region, married, car, save_act, current_act, mortgage e pep
categorical = ['sex', 'region', 'married', 'car', 'save_act', 'current_act', 'mortgage', 'pep']
bdNotDisc[categorical] = bdNotDisc[categorical].apply( LabelEncoder().fit_transform )
bdNotDisc = bdNotDisc.astype( 'int64' )
bdDisc[categorical] = bdDisc[categorical].apply( LabelEncoder().fit_transform )

# Prima analisi sulla distribuzione numerica dei dati attraverso histograms
histNotDisc = bdNotDisc.hist( bins=20, figsize=(20, 15), color='#E14906' )
plt.tight_layout(pad=4)
histDisc = bdDisc.hist( bins=20, figsize=(20, 15), color='#5BF58B' )
plt.tight_layout(pad=4)
plt.show()

# Calcolo della correlazione nel DB discretizzato
corr = bdDisc.corr()["pep"]
corr[np.argsort( corr, axis=0 )[::-1]]

# Plot della correlazione attraverso barh diagrams
num_feat = bdDisc.columns[bdDisc.dtypes != object]
num_feat = num_feat[0:11]
num_feat_noPep = num_feat[0:10]
labels = []
values = []
for col in num_feat_noPep:
    labels.append( col )
    values.append( np.corrcoef( bdDisc[col].values, bdDisc.pep.values )[0, 1] )

ind = np.arange( len( labels ) )
width = 0.9
fig, ax = plt.subplots( figsize=(12, 8) )
rects = ax.barh( ind, np.array( values ), color='green' )
ax.set_yticks( ind + ((width) / 2.) )
ax.set_yticklabels( labels, rotation='horizontal' )
ax.set_xlabel( "Coefficente di correlazione" )
ax.set_title( "Correlazione delle variabili con la features pep" );
plt.show()

# Matrice di correlazione di tutti gli attributi del dataframe
corrMatrix = bdDisc[num_feat].corr()
sns.set( font_scale=1.10 )
plt.figure( figsize=(10, 10) )
sns.heatmap( corrMatrix, vmax=.8, linewidths=0.01,
             square=True, annot=True, cmap='viridis', linecolor="white" )
plt.title( 'Correlation between features in the dataset' );
plt.show()

# Matrice di correlazione tra pep-income-children
corrMatrix = bdDisc[['pep', 'income', 'children']].corr()
sns.set( font_scale=1.10 )
plt.figure( figsize=(10, 10) )
sns.heatmap( corrMatrix, vmax=.8, linewidths=0.01,
             square=True, annot=True, cmap='viridis', linecolor="white" )
plt.title( 'Correlazione tra le features pep - income - children' );
plt.show()

# Preparazione dei dataset per i modelli, splitting di training e testing set
features = ['age', 'sex', 'region', 'income','married', 'children', 'car', 'save_act', 'current_act', 'mortgage']
target = 'pep'

# Features e target data per il dataset non discretizzato
x_data_not_disc = bdNotDisc[features]
y_data_not_disc = bdNotDisc[target]

# Features e target data per il dataset discretizzato
x_data_disc = bdDisc[features]
y_data_disc = bdDisc[target]

#Split in training e testing set di entrambi i dataset
X_train_disc, X_test_disc, y_train_disc, y_test_disc = train_test_split( x_data_disc, y_data_disc, test_size=0.3, random_state=0 )
X_train_not_disc, X_test_not_disc, y_train_not_disc, y_test_not_disc = train_test_split( x_data_not_disc, y_data_not_disc, test_size=0.3, random_state=0 )

# Classificatori utilizzati
RippersNames = [
    'RIPPER',
    'IREP',
]

Rippers = [
    lw.RIPPER(),
    lw.IREP(),
]

DecisionTreesNames = [
    'Decision Tree (Pruned max_len = 5)',
    'Decision Tree (Unpruned)',
]

DecisionTrees = [
    DecisionTreeClassifier(max_depth=5),
    DecisionTreeClassifier(),
]

NearestNeighborsNames = [
    'Nearest Neighbors (k=1)',
    'Nearest Neighbors (k=5)'
]

NearestNeighbors = [
    KNeighborsClassifier( 1 ),
    KNeighborsClassifier( 5 ),
]


# J48 sul dataset discretizzato e non discretizzato
for name, clf in zip( DecisionTreesNames, DecisionTrees ):
    print( name, ' sul modello non discretizzato' )
    clf.fit( X_train_not_disc, y_train_not_disc )
    y_pred = clf.predict( X_test_not_disc )
    print( accuracy_score( y_test_not_disc, y_pred ) )
    print( 'Accuracy: %0.2f\n' % accuracy_score( y_test_not_disc, y_pred ) )

    print( name, ' sul modello discretizzato' )
    clf.fit( X_train_disc, y_train_disc )
    y_pred = clf.predict( X_test_disc )
    print( accuracy_score( y_test_disc, y_pred ) )
    print( 'Accuracy: %0.2f\n' % accuracy_score( y_test_disc, y_pred ) )

# JRip sul dataset discretizzato e non discretizzato
for name, clf in zip( RippersNames, Rippers ):
    print( name, ' sul modello non discretizzato' )
    clf.fit( X_train_not_disc, y_train_not_disc )
    y_pred = clf.predict( X_test_not_disc )
    print( accuracy_score( y_test_not_disc, y_pred ) )
    print( 'Accuracy: %0.2f\n' % accuracy_score( y_test_not_disc, y_pred ) )

    print( name, ' sul modello discretizzato' )
    clf.fit( X_train_disc, y_train_disc )
    y_pred = clf.predict( X_test_disc )
    print( accuracy_score( y_test_disc, y_pred ) )
    print( 'Accuracy: %0.2f\n' % accuracy_score( y_test_disc, y_pred ) )


# IDk sul set non discretizzato
for name, clf in zip( NearestNeighborsNames, NearestNeighbors ):
    print( name, ' sul modello non discretizzato' )
    clf.fit( X_train_not_disc, y_train_not_disc )
    y_pred = clf.predict( X_test_not_disc )
    print( accuracy_score( y_test_not_disc, y_pred ) )
    print( 'Accuracy: %0.2f\n' % accuracy_score( y_test_not_disc, y_pred ) )


# Discretizzazione del campo income
bdDisc['income'] = pd.cut( x=bdDisc['income'],
                           bins=[5000, 11000, 17000, 23000, 29000, 35000, 41000, 47000, 53000, 59000, 65000],
                           labels=[11000, 17000, 23000, 29000, 35000, 41000, 47000, 53000, 59000, 65000] )

# Preparazione dei dataset per i modelli, splitting di training e testing set
x_data_disc = bdDisc[
    ['age', 'sex', 'region', 'income', 'children', 'car', 'save_act', 'current_act', 'mortgage', 'pep']]
y_data_disc = bdDisc['pep']

# Training e testing set per il modello discretizzato
X_train_disc, X_test_disc, y_train_disc, y_test_disc = train_test_split( x_data_disc, y_data_disc, test_size=0.3,
                                                                         random_state=0 )

# J48 sul dataset ulteriorimente discretizzato
for name, clf in zip( DecisionTreesNames, DecisionTrees ):
    print( name, ' sul modello ulteriormente discretizzato' )
    clf.fit( X_train_disc, y_train_disc )
    y_pred = clf.predict( X_test_disc )
    print( accuracy_score( y_test_disc, y_pred ) )
    print( 'Accuracy: %0.2f\n' % accuracy_score( y_test_disc, y_pred ) )

# IDk sul set ulteriormente discretizzato
for name, clf in zip( NearestNeighborsNames, NearestNeighbors ):
    print( name, ' sul modello ulteriormente discretizzato' )
    clf.fit( X_train_disc, y_train_disc )
    y_pred = clf.predict( X_test_disc )
    print( accuracy_score( y_test_disc, y_pred ) )
    print( 'Accuracy: %0.2f\n' % accuracy_score( y_test_disc, y_pred ) )

# Rippers sul set ulteriormente discretizzato
for name, clf in zip(RippersNames, Rippers):
    print(name, ' sul modello ulteriormente discretizzato')
    clf.fit(X_train_disc, y_train_disc)
    y_pred = clf.predict(X_test_disc)
    print(accuracy_score(y_test_disc, y_pred))
    print('Accuracy: %0.2f\n' % accuracy_score(y_test_disc, y_pred))

# Rappresentazione dei decision boundaries per il J48 pruned
clf = DecisionTreeClassifier(max_depth=5)
X = bdDisc[['income', 'age']].values
y = bdDisc['pep'].values

gs = gridspec.GridSpec(1, 1)
fig = plt.figure(figsize=(14, 10))
labels = ['Decision Tree pruned']

clf.fit(X, y)
fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
plt.title(labels)
plt.xlabel = 'Salary'
plt.ylabel = 'age'
plt.show()


# Plot dell'albero J48 not pruned
clf = DecisionTreeClassifier()
clf.fit(X, y)
plt.figure(figsize=(12, 12))
plot_tree( clf, fontsize=6 )
plt.savefig( 'tree_unpruned', dpi=100 )

# Plot dell'albero J48 pruned
clf = DecisionTreeClassifier( max_depth=5 )
clf.fit( X, y )
plt.figure( figsize=(12, 12) )
plot_tree( clf, fontsize=6 )
plt.savefig( 'tree_pruned', dpi=100 )