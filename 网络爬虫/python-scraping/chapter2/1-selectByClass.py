#from urllib.request import urlopen
from urllib import urlopen
from bs4 import BeautifulSoup

html = urlopen("http://www.pythonscraping.com/pages/warandpeace.html")
bsObj = BeautifulSoup(html, "html.parser")
# 指定html.parser解析器
nameList = bsObj.findAll("span", {"class":"green"})
# find()和findAll()通过标签的不同属性轻松过滤HTML页面
# nameList是列表形式的
for name in nameList:
# 遍历姓名列表
    print(name.get_text())
	# .get_text()方法将标签全都去掉，只剩下一串不带标签的文字
	# 应该在最后结果呈现时再用.get_text()
	