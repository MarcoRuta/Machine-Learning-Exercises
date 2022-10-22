import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from IPython.display import display

#Viene recuperato il dataset dal file .csv
data = pd.read_csv("online-retail-dataset.csv", sep=',')

#Rimozione degli spazi col metodo strip
data['Description'] = data['Description'].str.strip()

#Si selezionano i dati su cui manca il numero di fattura
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')

#Rimozione delle transazioni di credito
data = data[~data['InvoiceNo'].str.contains('C')]

#Si selezionano i dati relativi alle vendite in Francia
basket_France = (data[data['Country'] == "France"].groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))

#Si selezionano i dati relativi alle vendite negli UK
basket_UK = (data[data['Country'] == "United Kingdom"].groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))

def hot_encode(x):
    if (x <= 0) : return 0
    if (x >= 1) : return 1

# Encoding del datasets(tutti i valori negativi diventano 0, tutti i valori maggiori di 1 diventato 1)
basket_France = basket_France.applymap(hot_encode)
basket_UK = basket_UK.applymap(hot_encode)

#Generazione del modello con support almeno pari al 5%
frq_items = apriori(basket_France, min_support=0.05, use_colnames=True)

#Generazione delle regole con il supporto, la sicurezza e il lift corrispondente
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
rules.to_csv(r'C:\Users\lored\PycharmProjects\EserciziSIA_MarcoRuta\MarketBasket 17-05\Rules.csv', index = False)
rules.to_excel(r'C:\Users\lored\PycharmProjects\EserciziSIA_MarcoRuta\MarketBasket 17-05\Rules.xlsx')