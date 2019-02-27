
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
import matplotlib.pyplot as plt


x = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014']
height = [3265282, 7949783, 7738647, 6147242, 14758211, 16009954, 17639523, 20633976, 15987274, 15405719, 15833786, 
          14846184, 16277555, 16013951, 5439573]

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, height, color = 'blue', facecolor=None)
plt.xlabel("Years", color='black')
plt.ylabel("Size in words (tens of mlns)", color='black')

        
plt.xticks(x_pos, x, rotation=45)

plt.savefig('corpus_volume.png', bbox_inches = 'tight')
#plt.show()

