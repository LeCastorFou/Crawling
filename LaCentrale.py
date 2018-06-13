# coding: utf8
import urllib3
import bs4 as BeautifulSoup
import numpy as np
import pandas as pd
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

page = 1329
pageInit = page

while (T > 0):
    print('Page numer0 : ' + str(page) + ' Nb annonces ' + str(T))
    if (page == pageInit) :
        url = 'https://www.lacentrale.fr/listing?makesModelsCommercialNames=CITROEN&page=1329'
        #print(url)
    else:
        ptr = 'page='+str(page)
        #print(ptr)
        url2 = url.replace("page=1329",ptr)
        #print(url2)

    if(page == pageInit):
        response = http.request('GET', url)
    else:
        response = http.request('GET', url2)


    soup = BeautifulSoup.BeautifulSoup(response.data)

    allcont = soup.findAll('div',class_ = "adContainer ")
    T = len(allcont)

    Title = np.array(['mode','version','prix','km','annee','part','dpt','Gar'])

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

        # Pas toujour le meme format entre garantie et particulier
        p = 2
        g = 4
        #print(len(allcont[i].findAll('p')))
        #for j in range(0,len(allcont[i].findAll('p')) ):
            #print(allcont[i].findAll('p')[j] )

        if len(allcont[i].findAll('p')) == 3:
            p = 1
            g = 2
        if len(allcont[i].findAll('p')) == 4:
            p = 2
            g = 3
        if len(allcont[i].findAll('p')) == 2:
            p = 0
            g = 1
        # code particulier
        particulier = 0
        if allcont[i].findAll('p')[p].text == "Particulier":
            particulier = 1
        #departement
        dpt = allcont[i].find('div',class_ = "dptCont bold txtGrey7C lH35").text
        dpt = dpt.replace('Dpt. ','')

        # garantie en mois
        Gar = allcont[i].findAll('p')[g].text
        Gar = unicodedata.normalize('NFKD',Gar).encode('ascii','ignore')
        Gar= str(Gar)
        Gar =  re.sub("[^0-9]", "", Gar)

        if i == 0:
            annonce = np.array([model,version,prix,km,year,particulier,dpt,Gar])
        else:
            annonce = np.vstack((annonce,np.array([model,version,prix,km,year,particulier,dpt,Gar]) ))
    annonce = pd.DataFrame(annonce)
    annonce.columns = Title

    if(page == pageInit):
        Data = annonce
    else:
        Data = Data.append(annonce)

    page = page +1

    if page > 1999:
        break
    #if page == 4:
    #    T=1


Data.to_csv("CrawlerCitroen.csv", sep=',')
