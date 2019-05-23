# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:34:26 2019

@author: lindsey
"""

from bert_embedding import BertEmbedding
#import mxnet as mx
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib as mpl
import re
#example description for book#

masterfile=pd.read_csv("/home/lindsey/insightproject/bookdescriptionfromgoodreads.csv")

texttest=masterfile.iloc[0:30]
texttest.reset_index()
###################################################EMBEDDING STARTS###############################################################
m=768
i=0
#shaspe of bert vector is 768

avgs=pd.DataFrame()
avgsstacks=np.empty((1000,m))#initialize arrays for embeddings with 1000 (more than enough) sentences
for bo in np.arange(len(texttest)):##for every book
    currentbook=texttest.loc[texttest.index==bo]#extract one book at a time
    currentbook=currentbook.reset_index()
    booknumber=np.array(currentbook['Book#'])#book number
    description=str(np.array(currentbook['Description']))#book description
    sentences=description.split('.')#split book description into senteces by comma separation
#    ctx = mx.gpu(0)
    bert_embedding = BertEmbedding()
    result = bert_embedding(sentences)#convert book description into word embedding where each word is (768,)
    #average word embeddings
    for s in np.arange(len(result)-1):##last sentence is empty?
        words=(result[s])[1]#array is stored in the second element of the list
        avg=np.zeros(m,)#initiatlize matrix for stacking arrays of embeddings (number of sentences, 768)
        for w in np.arange(len(words)):#for each word embedding
            word=words[w]
            avg += word#add array element wise(768,)
        avg=avg/len(words)#averages over all words()
        current=pd.DataFrame( {'book#':booknumber, 'sentence#': s,'array':[avg],"index":i},index=[str(booknumber)])
        avgs=avgs.append(current,ignore_index=True) 
        avgsstacks[i,:]=avg
        i=i+1
avgsstacks=avgsstacks[0:len(avgs)]#getrid of empty rows of averages
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(avgsstacks)
x=new_values[:,0]
y=new_values[:,1]
len(x)==len(y)

#
def f2hex(f2rgb, f):
    rgb = f2rgb.to_rgba(f)[:3]#change colormap f2rgb to rgba color, where a is opacity
    return '#%02x%02x%02x' % tuple([int(255*fc) for fc in rgb])##scaling values from 0 to 1 back to 0 to 255
norm = colors.Normalize(vmin=0, vmax=max(np.array(avgs['book#'])))
f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('prism'))
#
for b in np.array(avgs['book#']):
    color=f2hex(f2rgb,b)
    avgs.loc[avgs['book#']==b,'color']=color    

#def save_text(text):
    
fig = plt.figure(figsize=(12.0, 7.0),dpi=100,linewidth=0.5)
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.5)

ax.set_xlabel('x',fontsize=11,linespacing=3.2)
ax.set_ylabel('y',fontsize=11,linespacing=3.2)

for i in range(len(x)):
    row=avgs.loc[avgs['index']==i]
    color=str(np.array(row['color'])).strip("['']")
    label=str(np.array(row['book#'])).strip("['']")
    a=ax.scatter(x[i],y[i], marker='o',c=str(color),s=100)
    ax.annotate(label,(x[i],y[i]))


plt.show()

