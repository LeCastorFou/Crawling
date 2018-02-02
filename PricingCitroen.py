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
data['mode'].unique()
len(data['version'].unique())


data_sub = data[data['mode']=="CITROEN DS4"]
plt.plot(data_sub['prix'],data_sub['km'],'ro')
plt.show()

plt.plot(data_sub['prix'],data_sub['annee'],'ro')
plt.show()

plotly.tools.set_credentials_file(username='valentin.lefranc', api_key='Xb0HU4LGnX8h3COhUgJr')

trace1 = go.Scatter(
    x = data_sub['prix'],
    y = data_sub['km'],
    mode='markers',
    marker=dict(size='16',color = data_sub['km'],showscale=True))
data = [trace1]


len(data_sub)

data_subtrain = data_sub[1:500]
data_subtest = data_sub[501:540]


############# Random RandomForest
features = data_subtrain.columns[[4,5]]
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
