from bs4 import BeautifulSoup
import requests
import re
import pandas as pd

def download_web(url):#amazon uses lxml    
    r=requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")##do unit text
    return soup
    
def find_description(soup):
    soup2=soup.find_all(id="descriptionContainer")## do unit text
    soupstr=str(soup2)
    text=soupstr.split("<a data-text-id")[0]
    text=text.split("freeTextContainer")[1]
    return text

#remove all tags and clean abit for goodreads
def remove_tags(text):
    tagswhere=re.compile('<.*?>')
    clean=re.sub(tagswhere,'',text)
    backwhere=re.compile('\n')
    cleaner=re.sub(backwhere,'',clean)
    final=cleaner.split(">")[1]
    return final

#
def make_frame(text,bookno,bookname):
    book=pd.DataFrame( {'Book#':bookno, "Book_Title":bookname,'Description':text},index=[str([i])])
    return book   

import numpy as np
booklist=pd.read_csv("/home/lindsey/Downloads/bookurl.csv")

urls=booklist['URL']
booknos=booklist['Book_#']
booknames=booklist['Book_Name']
texttest=pd.DataFrame()
for i in np.arange(len(urls)):
    url=urls[i]
    bookno=booknos[i]
    bookname=booknames[i]
    soup=download_web(url)
    text=find_description(soup)
    final=remove_tags(text)
    book=make_frame(final,bookno,bookname)
    texttest=texttest.append(book,ignore_index=True) 
    
texttest.to_csv("/home/lindsey/Desktop/bookdescriptionfromgoodreads.csv")
