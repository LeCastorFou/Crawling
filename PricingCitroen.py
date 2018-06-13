import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import datasets, linear_model
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

data = pd.read_csv("CrawlerCitroen.csv")
#data = pd.read_csv("Q:\ProjetsInternes\PricingVO\donnees\CrawlerCitroen.csv",encoding="utf-8")
#data['mode'].unique()
#len(data['version'].unique())
data = data.drop(data.columns[0], axis=1)
data = data.drop(data.columns[0], axis=1)

data_sub = data[data['mode']=="CITROEN DS4"]
plt.plot(data_sub['prix'],data_sub['km'],'ro')
plt.show()

plt.plot(data_sub['prix'],data_sub['annee'],'ro')
plt.show()

plotly.tools.set_credentials_file(username='valentin.lefranc', api_key='Xb0HU4LGnX8h3COhUgJr')

len(data_sub)
data_sub.columns
data_subtrain = data_sub[1:800]
data_subtest = data_sub[801:1032]


############# Random RandomForest
features = data_subtrain.columns[[3,4,5,7]]
y = data_subtrain['prix']
clf = RandomForestRegressor(n_estimators=20)
clf.fit(data_subtrain[features], y)
y_test = data_subtest['prix']

data_subtest['prediction'] = clf.predict(data_subtest[features])
data_subtest['error'] = (data_subtest['prix'] - clf.predict(data_subtest[features]))/data_subtest['prix']
hist = [go.Histogram(x=data_subtest['error'])]
py.iplot(hist, filename='basic histogram')

Emoy = np.sum( abs(data_subtest['error']) )/len(data_subtest.index)
Emoy

############# Regression linear
features = data_subtrain.columns[[4,5]]
y = data_subtrain['prix']
clf = linear_model.LinearRegression()
clf.fit(data_subtrain[features], y)
y_test = data_subtest['prix']

data_subtest['prediction'] = clf.predict(data_subtest[features])
data_subtest['error'] = (data_subtest['prix'] - clf.predict(data_subtest[features]))/data_subtest['prix']
hist = [go.Histogram(x=data_subtest['error'])]
py.iplot(hist, filename='basic histogram')

Emoy = np.sum( abs(data_subtest['error']) )/len(data_subtest.index)
Emoy


# Possible amélioration pour vehicules avec peu de km
# Voir l'impact de la version sur le prix
len(data['version'].unique())
Version = data.groupby('version').apply(len)
Version = pd.DataFrame(Version)
Version['version'] = Version.index
Version.columns = np.array(['no','version'])
Version = Version.sort_values(by =['no'], ascending=False)
Version = Version[Version['no']>10]
len(Version)
Version['version']

dataV = data[data['version'].isin(Version['version'] )]
dataV.columns

# On normalise le prix à model equivalent
def f(group):
    v = group['prix']
    group['prixnormModel'] = (v - v.min()) / (v.max() - v.min())
    return group

dataV = dataV.groupby('mode').apply(f)

sns.lmplot('prixnormModel','km', data=dataV, hue='annee', fit_reg=False)
plt.show()


### Impact de la version sur la DS4
dataVds4 = dataV[dataV['mode']=="CITROEN DS4"]
len(dataVds4)
sns.lmplot('prixnormModel','km', data=dataVds4, hue='version', fit_reg=False)
plt.show()

# dispersion du prix de la ds4
hist = [go.Histogram(x=dataVds4['prix'])]
py.iplot(hist, filename='basic histogram')


# On cherche le model le plus vendu avec le moteur le plus récurent
Modver = data.groupby(['mode','version']).apply(len)
Modver = pd.DataFrame(Modver)
Modver.columns = np.array(['no'])
Modver = Modver.sort_values(by =['no'], ascending=False)

Modver


data_max = data[data['mode']=="CITROEN C3 (3E GENERATION)"]
data_max = data_max[data_max['version']=="III 1.2 PURETECH 82 FEEL"]
len(data_max)
data_max

plt.plot(data_max['prix'],data_max['km'],'ro')
plt.show()

sns.lmplot('prix','km', data=data_max, hue='version', fit_reg=False)
plt.show()

## Mot clef pour les versions de C3
C3ver = np.array(data_max['version'])
keyw = C3ver[1].split(" ")
keyw[0]
len(C3ver)
for i in range(0,len(C3ver)):
    if i == 0:
        keyw = C3ver[i].split(" ")
    if i > 0:
        keyw = np.append(keyw,C3ver[i].split(" "))

np.unique(keyw)

## Mot clef pour les versions citroen
Nver = np.array(data['version'])
Allkeyw = Nver[1].split(" ")

for i in range(0,len(Nver)):
    if i == 0:
        Allkeyw = Nver[i].split(" ")
    if i > 0:
        Allkeyw = np.append(Allkeyw,Nver[i].split(" "))



Allkeyw = pd.DataFrame(Allkeyw)
Allkeyw.columns = np.array(['kw'])
Allkeyw_S = Allkeyw.groupby(['kw']).apply(len)
Allkeyw_S = pd.DataFrame(Allkeyw_S)
Allkeyw_S.columns = np.array(['oc'])
Allkeyw_S = Allkeyw_S.sort_values(by =['oc'], ascending=False)
# On ne garde que les mots clefs qui apparaissent plus d'une fois sur 100
Allkeyw_S = Allkeyw_S[Allkeyw_S['oc'] > 0.01*len(Allkeyw)]
Allkeyw_S.index[2]

data['kw'] = 1
a = np.linspace(0,0,len(Allkeyw_S))

for j in range(0,len(Allkeyw_S)):
    for i in range(0,len(data['version'])):
         if np.array(data['version'])[i].find(Allkeyw_S.index[j])   == -1:
             b=1
         if np.array(data['version'])[i].find(Allkeyw_S.index[j])   != -1:
            a[j] = a[j] + np.array(data['version'])[i].find(Allkeyw_S.index[j])

    a[j] = a[j]/np.array(Allkeyw_S)[j]


kwdf = pd.DataFrame(np.array(Allkeyw_S.index))
kwdf['pos'] = a
kwdf = kwdf.sort_values(by =['pos'], ascending=True)
kwdf['type'] = "Uk"
kwdf['type'][kwdf['pos']<1] = "Nmod"
kwdf['type'][kwdf['pos']>3 ] = "litre"
kwdf['type'][kwdf['pos']>5 ] = "moteur"
kwdf['type'][kwdf['pos']>10] = "PUI"
kwdf['type'][kwdf['pos']>17] = "OPT"

data['litre'] = 1
litre = kwdf[kwdf['type'] == "litre"]
litre.columns = np.array(['litre','pos','type'])

for j in range(0,len(litre)):
    print(np.array(litre['litre'])[j])
    for i in range(0,len(data['version'])):
        if np.array(data['version'])[i].find( np.array(litre['litre'])[j] )   != -1:
             data['litre'][data.index[i]] = np.array(litre['litre'])[j]

data['moteur'] = 'moteur'
moteur = kwdf[kwdf['type'] == "moteur"]
moteur.columns = np.array(['moteur','pos','type'])
moteur

for j in range(0,len(moteur)):
    print(np.array(moteur['moteur'])[j])
    for i in range(0,len(data['version'])):
        if np.array(data['version'])[i].find( np.array(moteur['moteur'])[j] )   != -1:
             data['moteur'][data.index[i]] = np.array(moteur['moteur'])[j]

data_f = data
data_f = data_f.replace({'E-HDI': 1}, regex=True)
data_f = data_f.replace({'BLUEHDI': 2}, regex=True)
data_f = data_f.replace({'PURETECH': 3}, regex=True)
data_f = data_f.replace({'moteur': 0}, regex=True)

data_f


data_sub = data_f[data_f['mode']=="CITROEN DS4"]
len(data_sub)
data_sub.columns

data_subtrain = data_sub[1:800]
data_subtest = data_sub[801:1032]

############# Random RandomForest
features = data_subtrain.columns[[3,4,5,7,9,10]]
y = data_subtrain['prix']
clf = RandomForestRegressor(n_estimators=20)
clf.fit(data_subtrain[features], y)
y_test = data_subtest['prix']

data_subtest['prediction'] = clf.predict(data_subtest[features])
data_subtest['error'] = (data_subtest['prix'] - clf.predict(data_subtest[features]))/data_subtest['prix']
hist = [go.Histogram(x=data_subtest['error'])]
py.iplot(hist, filename='basic histogram')

Emoy = np.sum( abs(data_subtest['error']) )/len(data_subtest.index)
Emoy

############# Random RandomForest c3 PURETECH
data_max.columns
data_subtrain = data_max[1:400]
data_subtest = data_max[401:574]

features = data_subtrain.columns[[3,4,5,7]]
y = data_subtrain['prix']
clf = RandomForestRegressor(n_estimators=20)
clf.fit(data_subtrain[features], y)
y_test = data_subtest['prix']

data_subtest['prediction'] = clf.predict(data_subtest[features])
data_subtest['error'] = (data_subtest['prix'] - clf.predict(data_subtest[features]))/data_subtest['prix']
hist = [go.Histogram(x=data_subtest['error'])]
py.iplot(hist, filename='basic histogram')

Emoy = np.sum( abs(data_subtest['error']) )/len(data_subtest.index)
Emoy
