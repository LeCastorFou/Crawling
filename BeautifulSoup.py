import urllib3
import bs4 as BeautifulSoup

http = urllib3.PoolManager()

url = 'http://www.d8.tv/d8-series/pid6654-d8-longmire.html'
response = http.request('GET', url)

soup = BeautifulSoup.BeautifulSoup(response.data)

print(soup.prettify())

soup.head.meta

soup.meta.find_next_sibling().parent
soup.meta.find_next_sibling().parent.parent

soup.find('div')
soup.find('div',class_ = "nIHRFAQHImBAVR7Fn_y4F")
type(soup.find('div',attrs={"class":u"nIHRFAQHImBAVR7Fn_y4F"}))
soup.find('div',attrs={"class":u"nIHRFAQHImBAVR7Fn_y4F"})



import urllib3
import bs4 as BeautifulSoup
import numpy as np
import pandas as pd
import re
import unicodedata


http = urllib3.PoolManager()


url = 'https://www.lacentrale.fr/listing?makesModelsCommercialNames=CITROEN&page=1'

response = http.request('GET', url)
soup = BeautifulSoup.BeautifulSoup(response.data)
allcont = soup.findAll('div',class_ = "adContainer ")

model = allcont[1].span.text
version = allcont[1].findAll('span')[1].text

for i in range(0,15):
    a = allcont[i].find('span',class_ = "txtGrey3").text
    print(i)
    if type(allcont[i].find('span',class_ = "version txtGrey7C noBold")) ==  type(None):
        print('is none type')
        b =allcont[i].findAll('span')[3].text
    else :
        print('is ok')
        b = allcont[i].find('span',class_ = "version txtGrey7C noBold").text
    print(a + " " + b)


allcont[i].findAll('span')[3].text



type(allcont[0].find('span',class_ = "version txtGrey7C noBold"))

allcont[1].find('span',class_ = "version txtGrey7C noBold").text


allcont
for i in range(0,15):
    print(allcont[i].find('p',class_ ="txtBlack typeSeller hiddenPhone").text)
    Gar = allcont[i].find('div',class_ ="warranty bold hiddenPhone").text
    Gar = unicodedata.normalize('NFKD',Gar).encode('ascii','ignore')
    Gar= str(Gar)
    Gar =  re.sub("[^0-9]", "", Gar)
    print(Gar)
    
