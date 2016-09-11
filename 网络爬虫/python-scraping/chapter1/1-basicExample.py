# from urllib.request import urlopen
from urllib import urlopen
#Retrieve HTML string from the URL
html = urlopen("http://www.pythonscraping.com/exercises/exercise1.html")
print(html.read())