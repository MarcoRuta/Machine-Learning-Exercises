import pandas as pd
import matplotlib.pyplot as plt
#Viene recuperato il dataset dal file .csv
csv = pd.read_csv('PostiLetto.csv', sep=';')

print (csv.head())
print (csv.dtypes)

#Vengono convertiti in Int64 i valori Obj (causati dai N.D.)
csv.convert_dtypes(convert_integer=True )


print (csv.dtypes)

#Si selezionano i dati relativi solo al 2014
csv2014 = csv[ csv['Anno'] == 2014]
beds2014 = csv2014['Totale posti letto']

print(beds2014.describe())

#Si crea un istogramma relativo alla distribuzione numero posti letto / numero di ospedali
histogram = beds2014.hist( bins = 50)
histogram.set_title('Distribuzione posti letto - ospedali 2014')
histogram.set_xlabel('Numero di posti letto')
histogram.set_ylabel('Ospedali')

plt.show()

#Si ordinano i valori trovati in base al numero di posti letto
csv2014 = csv2014.sort_values('Totale posti letto', ascending = False)

print(csv2014[['Denominazione Struttura', 'Totale posti letto']])

#Si recupera il numero di letti per regione nel 2014
bedsByRegion = csv2014[['Descrizione Regione','Totale posti letto']].groupby(['Descrizione Regione'])

#Si effettua il sort
summedAndSortedBedsByRegion = bedsByRegion.sum().sort_values('Totale posti letto')

#Viene stampato il diagramma a barre
summedAndSortedBedsByRegion.plot.bar()

plt.show()
