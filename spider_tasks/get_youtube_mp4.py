#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2021/3/9 6:35 下午
@File    : get_youtube_mp4.py
@Desc    : 

"""


import requests
from requests.exceptions import ReadTimeout, ConnectTimeout, HTTPError, ConnectionError, RequestException
from lxml import etree
import json,hashlib,re
import random,time
import json

url = "https://www.youtube.com/results?search_query=jk制服+抖音快手"
user_agents =[
    {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; rv:22.0) Gecko/20130405 Firefox/22.0"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; rv:14.0) Gecko/20100101 Firefox/18.0.1"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.2309.372 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:21.0) Gecko/20130331 Firefox/21.0"},
    {"User-Agent": "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)"},
    {"User-Agent": "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)"},
    {"User-Agent": "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)"},
    {"User-Agent": "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)"},
    {"User-Agent": "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)"},
    {"User-Agent": "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)"},
    {"User-Agent": "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)"},
    {"User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)"},
    {"User-Agent": "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6"},
    {"User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1"},
    {"User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0"},
    {"User-Agent": "Mozilla/5.0(Macintosh;U;IntelMacOSX10_6_8;en-us)AppleWebKit/534.50(KHTML,likeGecko)Version/5.1Safari/534.50"},
    {"User-Agent": "Mozilla/5.0(Windows;U;WindowsNT6.1;en-us)AppleWebKit/534.50(KHTML,likeGecko)Version/5.1Safari/534.50"},
    {"User-Agent": "Mozilla/5.0(compatible;MSIE9.0;WindowsNT6.1;Trident/5.0"},
    {"User-Agent": "Mozilla/4.0(compatible;MSIE8.0;WindowsNT6.0;Trident/4.0)"},
    {"User-Agent": "Mozilla/4.0(compatible;MSIE7.0;WindowsNT6.0)"},
    {"User-Agent": "Mozilla/4.0(compatible;MSIE6.0;WindowsNT5.1)"},
    {"User-Agent": "Mozilla/5.0(Macintosh;IntelMacOSX10.6;rv:2.0.1)Gecko/20100101Firefox/4.0.1"},
    {"User-Agent": "Mozilla/5.0(WindowsNT6.1;rv:2.0.1)Gecko/20100101Firefox/4.0.1"},
    {"User-Agent": "Opera/9.80(Macintosh;IntelMacOSX10.6.8;U;en)Presto/2.8.131Version/11.11"},
    {"User-Agent": "Opera/9.80(WindowsNT6.1;U;en)Presto/2.8.131Version/11.11"},
    {"User-Agent": "Mozilla/5.0(Macintosh;IntelMacOSX10_7_0)AppleWebKit/535.11(KHTML,likeGecko)Chrome/17.0.963.56Safari/535.11"},
    {"User-Agent": "Mozilla/4.0(compatible;MSIE7.0;WindowsNT5.1;Maxthon2.0)"},
    {"User-Agent": "Mozilla/4.0(compatible;MSIE7.0;WindowsNT5.1;TencentTraveler4.0)"},
    {"User-Agent": "Mozilla/4.0(compatible;MSIE7.0;WindowsNT5.1)"},
    {"User-Agent": "Mozilla/4.0(compatible;MSIE7.0;WindowsNT5.1;TheWorld)"},
    {"User-Agent": "Mozilla/4.0(compatible;MSIE7.0;WindowsNT5.1;Trident/4.0;SE2.XMetaSr1.0;SE2.XMetaSr1.0;.NETCLR2.0.50727;SE2.XMetaSr1.0)"},
    {"User-Agent": "Mozilla/4.0(compatible;MSIE7.0;WindowsNT5.1;360SE)"},
    {"User-Agent": "Mozilla/4.0(compatible;MSIE7.0;WindowsNT5.1)"},
    {"User-Agent": "Mozilla/4.0(compatible;MSIE7.0;WindowsNT5.1;AvantBrowser)"},
    {"User-Agent": "Mozilla/5.0(iPhone;U;CPUiPhoneOS4_3_3likeMacOSX;en-us)AppleWebKit/533.17.9(KHTML,likeGecko)Version/5.0.2Mobile/8J2Safari/6533.18.5"},
    {"User-Agent": "Mozilla/5.0(iPod;U;CPUiPhoneOS4_3_3likeMacOSX;en-us)AppleWebKit/533.17.9(KHTML,likeGecko)Version/5.0.2Mobile/8J2Safari/6533.18.5"},
    {"User-Agent": "Mozilla/5.0(iPad;U;CPUOS4_3_3likeMacOSX;en-us)AppleWebKit/533.17.9(KHTML,likeGecko)Version/5.0.2Mobile/8J2Safari/6533.18.5"},
    {"User-Agent": "Mozilla/5.0(Linux;U;Android2.3.7;en-us;NexusOneBuild/FRF91)AppleWebKit/533.1(KHTML,likeGecko)Version/4.0MobileSafari/533.1"},
    {"User-Agent": "MQQBrowser/26Mozilla/5.0(Linux;U;Android2.3.7;zh-cn;MB200Build/GRJ22;CyanogenMod-7)AppleWebKit/533.1(KHTML,likeGecko)Version/4.0MobileSafari/533.1"},
    {"User-Agent": "Opera/9.80(Android2.3.4;Linux;OperaMobi/build-1107180945;U;en-GB)Presto/2.8.149Version/11.10"},
    {"User-Agent": "Mozilla/5.0(Linux;U;Android3.0;en-us;XoomBuild/HRI39)AppleWebKit/534.13(KHTML,likeGecko)Version/4.0Safari/534.13"},
    {"User-Agent": "Mozilla/5.0(BlackBerry;U;BlackBerry9800;en)AppleWebKit/534.1+(KHTML,likeGecko)Version/6.0.0.337MobileSafari/534.1+"},
    {"User-Agent": "Mozilla/5.0(hp-tablet;Linux;hpwOS/3.0.0;U;en-US)AppleWebKit/534.6(KHTML,likeGecko)wOSBrowser/233.70Safari/534.6TouchPad/1.0"},
    {"User-Agent": "Mozilla/5.0(SymbianOS/9.4;Series60/5.0NokiaN97-1/20.0.019;Profile/MIDP-2.1Configuration/CLDC-1.1)AppleWebKit/525(KHTML,likeGecko)BrowserNG/7.1.18124"},
    {"User-Agent": "Mozilla/5.0(compatible;MSIE9.0;WindowsPhoneOS7.5;Trident/5.0;IEMobile/9.0;HTC;Titan)"}
]

INFO_XPATH = '''.//div[@class='style-scope ytd-app']'''
headers = random.choice(user_agents)
res = requests.get(url, headers=headers)

if res.status_code == 200:
    jsonData = res.text
    print(jsonData)
    pt = etree.HTML(jsonData.lower(), parser=etree.HTMLParser(encoding='utf-8'))
    tb = pt.xpath(INFO_XPATH)
    print(tb)
    for x in tb:
        print(x)

