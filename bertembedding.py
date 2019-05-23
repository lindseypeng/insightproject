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




#shaspe of bert vector is 768

booknumber=1






sentences = book_1.split('.')
bert_embedding = BertEmbedding()
result = bert_embedding(sentences)
m=768
#initialize the average word vector
avgs=pd.DataFrame()
i=0
avgsstack=np.empty((len(result)-1,m))
for s in np.arange(len(result)-1):##last sentence is empty?
    words=(result[s])[1]
    avg=np.zeros(m,)
    ##initiatlize matrix for stacking arrays of embeddings (number of sentences, 768)
    for w in np.arange(len(words)):
        ##keep track of total number of rows
        word=words[w]
        avg += word
    avg=avg/len(words)
    current=pd.DataFrame( {'book#':booknumber, 'sentence#': s,'array':[avg],"index":i},index=[str(booknumber)])
    avgs=avgs.append(current,ignore_index=True)  
    avgsstack[s,:]=avg
    i=i+1



#Creates and TSNE model and plots it"


#norm = colors.Normalize(vmin=0, vmax=max(np.array(avgs['book#'])))
norm = colors.Normalize(vmin=0, vmax=10)
f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('RdYlGn'))

def f2hex(f2rgb, f):
    rgb = f2rgb.to_rgba(f)[:3]#change colormap f2rgb to rgba color, where a is opacity
    return '#%02x%02x%02x' % tuple([int(255*fc) for fc in rgb])##scaling values from 0 to 1 back to 0 to 255
    
for b in np.array(avgs['book#']):
    color=f2hex(f2rgb,b)
    avgs.loc[avgs['book#']==b,'color']=color



###
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(avgsstack)##converts 768 dim to 2 dim, (number of sentences, 768)
##record x and y position of the 2d array embeddgins
x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])
    
plt.figure(figsize=(16, 16)) 
for i in range(len(x)):
    plt.scatter(x[i],y[i])
    row=avgs.loc[avgs.index==i]
    label="book{} sentence{}".format(str(np.array(row['book#'])),str(np.array(row['sentence#'])))
    color=np.array(row['color'])
    plt.annotate(label,xy=(x[i], y[i]),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom',c=str(color),s=100)
plt.show()

