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
