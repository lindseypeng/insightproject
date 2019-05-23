# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:34:26 2019

@author: lindsey
"""

from bert_embedding import BertEmbedding
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib as mpl
#example description for book#
book_1 = """In a series of engaging and insightful examples—from the breeding of hens to the timing of \
cataract surgeries to the organization of an automobile plant—Wilson shows how an evolutionary worldview provides a practical\
tool kit for understanding not only genetic evolution but also the fast-paced changes that are having an impact on our world and ourselves. \
What emerges is an incredibly empowering argument: If we can become wise managers of evolutionary processes, we can solve the problems of our \
age at all scales—from the efficacy of our groups to our well-being as individuals to our stewardship of the planet Earth."""
book_2 = """Fascinated by aging and mortality, West applied the rigor of a physicist to the biological question \
of why we live as long as we do and no longer. The result was astonishing, and changed science: West found that despite \
the riotous diversity in mammals, they are all, to a large degree, scaled versions of each other. If you know the size of a mammal,\
 you can use scaling laws to learn everything from how much food it eats per day, what its heart-rate is, how long it will take to\
 mature, its lifespan, and so on."""
book_3="""The science of emotion is in the midst of a revolution on par with the discovery of relativity in physics\
 and natural selection in biology. Leading the charge is psychologist and neuroscientist Lisa Feldman Barrett, whose research\
 overturns the long-standing belief that emotions are automatic, universal, and hardwired in different brain regions. \
Instead, Barrett shows, we construct each instance of emotion through a unique interplay of brain, body, and culture."""



booknumbers=[1,2,3]
bookparagraphs=[book_1,book_2,book_3]
texttest=pd.DataFrame()
for i in [0,1,2]:
    book=pd.DataFrame( {'Book#':booknumbers[i], 'Description':bookparagraphs[i]},index=[str(booknumbers[i])])
    texttest=texttest.append(book,ignore_index=True)  

###################################################EMBEDDING STARTS###############################################################
    
m=768
i=0
#shaspe of bert vector is 768
avgs=pd.DataFrame()
avgsstacks=np.empty((1000,m))#initialize arrays for embeddings with 1000 (more than enough) sentences
for bo in np.arange(len(texttest)):##for every book
    currentbook=texttest.loc[texttest.index==bo]#extract one book at a time
    booknumber=np.array(currentbook['Book#'])#book number
    description=str(np.array(currentbook['Description']))#book description
    sentences=description.split('.')#split book description into senteces by comma separation
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
f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('RdYlGn'))
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
#cb1=mpl.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,orientation='vertical')
for i in range(len(x)):
    row=avgs.loc[avgs['index']==i]
    color=str(np.array(row['color'])).strip("['']")
    label=str(np.array(row['book#'])).strip("['']")
    a=ax.scatter(x[i],y[i], marker='o',c=str(color),s=100)
    ax.annotate(label,(x[i],y[i]))

#plt.colorbar()
plt.show()
