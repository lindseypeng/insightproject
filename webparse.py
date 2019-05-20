# -*- coding: utf-8 -*-
"""
Created on Sun May 19 19:27:00 2019

@author: lindsey
"""



url="https://www.amazon.ca/This-View-Life-Completing-Revolution/dp/1101870206?pf_rd_p=6d4e5a04-483b-427e-8f01-aa9c5a81e67a&pd_rd_wg=wtuut&pf_rd_r=V4266G3A0ABRFA3QFMTZ&ref_=pd_gw_cr_simh&pd_rd_w=876Ax&pd_rd_r=fe73f8f7-4e6b-4094-9b7c-2836175ed3da"
#bookname=""
##extract bookdescription from amazon##
def find_book_description(url,parsetype):#amazon uses lxml
    from bs4 import BeautifulSoup
    import requests
    r=requests.get(url)
    soup=BeautifulSoup(r.text,parsetype)
    soup=soup.find_all(id="bookDescription_feature_div")
    soupstr=str(soup)
    text=soupstr.split("</div>")[0]
    text=text.split("<noscript>")[1]
    return text

#remove all tags and clean abit
def remove_tags(text):
    import re
    tagswhere=re.compile('<.*?>')
    clean=re.sub(tagswhere,'',text)
    backwhere=re.compile('\n')
    cleaner=re.sub(backwhere,'',clean)
    return cleaner

#
#def save_text(text):
    
    