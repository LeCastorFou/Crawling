# coding: utf8
import urllib3
import bs4 as BeautifulSoup
import numpy as np
import pandas as pd
from unidecode import unidecode
import encodings
import codecs
import re
import unicodedata

#type(soup.find('div',attrs={"class":u"nIHRFAQHImBAVR7Fn_y4F"}))
#soup.find('div',attrs={"class":u"nIHRFAQHImBAVR7Fn_y4F"})
#ann = rech.find('div',class_ = "adContainer ")
#model = ann.find('span',class_ = "txtGrey3")
#price = rech.find('div',class_ = "fieldPrice")
#km = rech.find('div',class_ = "fieldMileage")
#year = rech.find('div',class_ = "fieldYear")

#km =km.text
#km
#year = year.text
#str(model)
#ann.find_all('span',class_ = "txtGrey3")
#print(soup.find('div',class_ = "adContainer ").prettify() )

# voir find_all

#rech = soup.find('div',class_ = "resultList mB15 hiddenOverflow listing ")



http = urllib3.PoolManager()

T = 16
page = 1

while (T == 16):
    print('Page numer0 : ' + str(page) + ' Nb annonces ' + str(T))
    if (page == 1) :
        url = 'https://www.lacentrale.fr/listing?makesModelsCommercialNames=CITROEN&page=1'
        #print(url)
    else:
        ptr = 'page='+str(page)
        #print(ptr)
        url2 = url.replace("page=1",ptr)
        #print(url2)

    if(page == 1):
        response = http.request('GET', url)
    else:
        response = http.request('GET', url2)


    soup = BeautifulSoup.BeautifulSoup(response.data)

    allcont = soup.findAll('div',class_ = "adContainer ")
    #dir(allcont[0])
    T =len(allcont)
    Title = np.array(['mode','version','prix','km','annee'])

    for i in range(0,T):
        model = allcont[i].span.text
        version = allcont[i].findAll('span')[1].text
        prix = allcont[i].find('div',class_ = "fieldPrice").text
        km = allcont[i].find('div',class_ = "fieldMileage").text
        year = allcont[i].find('div',class_ = "fieldYear").text
        prix = unicodedata.normalize('NFKD',prix).encode('ascii','ignore')
        prix = str(prix)
        prix = re.sub("[^0-9]", "", prix)
        km = unicodedata.normalize('NFKD',km).encode('ascii','ignore')
        km = str(km)
        km = re.sub("[^0-9]", "", km)
        if i == 0:
            annonce = np.array([model,version,prix,km,year])
        else:
            annonce = np.vstack((annonce,np.array([model,version,prix,km,year]) ))
    annonce = pd.DataFrame(annonce)
    annonce.columns = Title

    if(page == 1):
        Data = annonce
    else:
        Data = Data.append(annonce)

    page = page +1


Data.to_csv("CrawlerCitroen.csv", sep=',')
