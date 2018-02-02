import urllib3
import bs4 as BeautifulSoup

http = urllib3.PoolManager()

url = 'https://www.lacentrale.fr/occasion-voiture-marque-citroen.html'
response = http.request('GET', url)

soup = BeautifulSoup.BeautifulSoup(response.data)

# voir find_all

rech =soup.find('div',class_ = "resultList mB15 hiddenOverflow listing ")
#type(soup.find('div',attrs={"class":u"nIHRFAQHImBAVR7Fn_y4F"}))
#soup.find('div',attrs={"class":u"nIHRFAQHImBAVR7Fn_y4F"})
ann = rech.find('div',class_ = "adContainer ")
model = ann.find('span',class_ = "txtGrey3")
price = rech.find('div',class_ = "fieldPrice")
km = rech.find('div',class_ = "fieldMileage")
year = rech.find('div',class_ = "fieldYear")

km =km.text
km
year = year.text

str(model)
