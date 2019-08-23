#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-8-22 下午6:16
@File    : gre_spider.py
@Desc    : gre 爬虫
"""


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait  # available since 2.4.0
from selenium.webdriver.support import expected_conditions as EC  # available since 2.26.0
from selenium.webdriver.phantomjs.webdriver import WebDriver

# import requests
#
log_url = "https://gre.etest.net.cn/login.do"
neea_id = "71548689"
password = "ZouShuai_92"
# logo_data = {"neeaId": "71548689", "password": "ZouShuai_92",
#             "checkImageCode": "judm", "CSRFToken": "7feec5b4-fb7f-466d-8cec-b9595ec485d1"}
#
#
# url = "https://gre.etest.net.cn/testSites.do?p=testSites&m=ajax&ym=2019-09&neeaID=71548689&cities=BEIJING_BEIJING%3BTIANJIN_TIANJIN%3BHEBEI_SHIJIAZHUANG%3B&citiesNames=%25E5%258C%2597%25E4%25BA%25AC%3B%25E5%25A4%25A9%25E6%25B4%25A5%3B%25E7%259F%25B3%25E5%25AE%25B6%25E5%25BA%2584%3B&whichFirst=AS&isFilter=0&isSearch=1&MmEwMD=47N.aDRijSU9vBgdZDb3Z9BglFe77TiiTB7MzGZgWAkWVRPW_QmIEa9in64CPCT4NN4gS_MFG8ardihKMAhpszdphU7LjwALPx5w8MvqV3aLLM9ekA7i78aotrjgH9ymgISO2xKkIr9wC2L7HkEoIg76o6OONR3T0waihzQPy_BGq8LYDiGZNCSiddhs2iSzA_Jld7q1Q9yM8_1hcNPO5OwcsbQ8Xu.VxJDHJc0lTwJGwtFkUJgTdmHad86Zp2Uo62AafFwfQmfIw4rhRYrMnSFvuI9iwLhlwtnjPu4wQ96Y191.fTxhhf1z1tBr58aFXg.rYzgN.FtS5oLzmlDr3OVz.yYVyZZja4oSWs20DwCRTLqV_.qZCMgyyuVjEUk0erna.ZDwfCyoDvU85l4Ewzr4wA5XO5PSEUxYnb8r.oEW6oC"
# header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
#           'Cookie': 'BIGipServerhw_gre_pool=319473674.22784.0000; FSSBBIl1UgzbN7N443S=1TSNaEOZZI8dygSWpzrgOhjuKWcho6lm5p6.zT0.lGYbgxRX76T2zI5q33eIGJLn; BIGipServerHW-ruishu-cache=2854144010.20480.0000; ajaxStep=myHome; JSESSIONID=80C183B329CEA3B06396D33610149D92; step=myStatus.do; FSSBBIl1UgzbN7N443T=4nXtE38B1BAmISd8l3fyl_LRZl_JV.xBJS._Y54R0ilz7o2zccc5aXOBtNZUFLnFp6ZRx9Q43q6GoAW6OiWa.ugauHXDSa6AXqxBVkCZZ2kjARa6ORBFhmGpDPoxTshQO18nbdCTxrQ.scZ195fLpdpcWyDOhFOpHX1GP4hsFXbU18nsOsUJWQ2t.fVtHItbI9PP0QejP0RTZwdDv2YXFQHso8yAA8ah62ARX0ir2XOb0rFq32o5eRzSbhHOH688Z7CzpM8lAGwkBr1g3_8j5tRaw4EXfn19KR2RVUFjZ7tV2eHVulX3vmiFnjerNMAOcffQKds7MqpFlqvMl57yClyj0yaO3NBZZOZcGlbX1SX_Wla'}
# data = {'p': 'testSites', 'm': 'ajax', 'ym': '2019-09', 'neeaID': 71548689, 'cities': 'BEIJING_BEIJING;TIANJIN_TIANJIN;HEBEI_SHIJIAZHUANG;', 'citiesNames': '%E5%8C%97%E4%BA%AC;%E5%A4%A9%E6%B4%A5;%E7%9F%B3%E5%AE%B6%E5%BA%84;', 'whichFirst': 'AS', 'isFilter': 0, 'isSearch': 1, 'MmEwMD': '41CWxgDNsHEk8pv3mgydmFaIe9PhYhpNWpuZV8OITXQ.zve.kPFgSi4Nhx9pMEjQAU9IElleRGBbDaVSPXVCjsGCnccF.XHTV0JcpPbkxQbwtHUpGkREuJDnMawJQXpfCOZodB.lkKJwtsejAARx_YU_CGb2hTAgHDeshbkm7pomQlr4DmmNbV2CuDVIHnDb842eCrPcHU3XKNcuiyLUl4uG9XHhgI9CCGUa2ADHqVgc2RtsMWgLDVuW70mR.oTVNP33TsGoShXxvP3YSM6vBWeckHIid.Puifr6qVLUQk7OyL8.e.LBBJPr5mzrgCBEQ8wfYdklm1u_ZomyC1ZAHx1iWDPYz0nD87tssejFPekhRHq8gpOWspDnPDPFwvFQiVvupexqFk._38rfpGZ3uThhKe.igNDr_LYlMVh9O9YU.Qp'}
#
#
# session = requests.Session()
# # session.post(log_url, headers=header, data=logo_data)
# response = session.get(url, headers=header)
# print(response.status_code)
# print(response.text)




# Create a new instance of the phantomjs driver
options = Options()
# options.add_argument('--headless')
# 设置成用户自己的数据目录
# options.add_argument('--user-data-dir=~/.config/google-chrome/Default')
# 修改浏览器的User-Agent来伪装你的浏览器访问手机m站
# options.add_argument('--user-agent=iphone')
# 浏览器启动时安装crx扩展
# options.add_extension('/path/to/xxx.crx')  # 自己下载的crx路径

driver = webdriver.Firefox(firefox_options=options)
# driver = webdriver.Chrome(chrome_options=options)
# driver = WebDriver(executable_path='/opt/phantomjs-2.1.1-linux-x86_64/bin/phantomjs', port=5001)
wait = WebDriverWait(driver, 5)
driver.get(log_url)
print(driver.title)

login = driver.find_element_by_id("neeaId")
login.clear()
login.send_keys(neea_id)
login.submit()
pwd = driver.find_element_by_id("password")
pwd.clear()
pwd.send_keys(password)
pwd.submit()
driver.find_element_by_id("checkImageCode").click()
wait.until(lambda driver: driver.find_element_by_id('checkImageCode'))
image_code = input()
login_check = driver.find_element_by_id("checkImageCode")
login_check.clear()
login_check.send_keys(image_code)
login_check.submit()

driver.find_element_by_class_name("btn btn-primary").click()


# inputElement = driver.find_element_by_id("kw")
# inputElement.send_keys("cheese!")

# inputElement.submit()

# try:
#     # we have to wait for the page to refresh, the last thing that seems to be updated is the title
#     WebDriverWait(driver, 10).until(EC.title_contains("cheese!"))
#
#     # You should see "cheese! - Google Search"
#     print(driver.title)
#     print(driver.get_cookies())
#
# finally:
#     driver.quit()


