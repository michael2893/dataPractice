#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


data = np.random.randn(2,3
                      )


# In[3]:


data


# In[4]:


data * 10


# In[5]:


data + data


# In[6]:


data1 = [6.,7.5,8.,0.,1.]


# In[7]:


arr1 = np.array(data1)


# In[8]:


arr1


# In[9]:


data2 = [[1,2,3,4],[5,6,7,8]]


# In[10]:


arr2=np.array(data2
            )


# In[11]:


arr2


# In[12]:


arr2.ndim


# In[13]:


arr2.shape


# In[14]:


arr1.dtype


# In[15]:


arr2.dtype


# In[16]:


np.zeros(10)


# In[17]:


np.zeros((3,6))


# In[18]:


np.empty((2,3,2))


# In[19]:


np.arrange(!5)


# In[20]:


np.arrange(15)


# In[21]:


np.ones_like


# In[22]:


arr1= np.array([1,2,3], dtype= np.int32)


# In[23]:


arr1


# In[24]:


arr1*arr2


# In[2]:


import numpy as np


# In[5]:


array = np.arange(10)


# In[6]:


arr


# In[7]:


array


# In[8]:


array[5]


# In[9]:


array[5:8]


# In[10]:


array[5:8] = 12


# In[11]:


array


# In[12]:


array_slice=array[5:8]


# In[13]:


array_sice


# In[14]:


array_slice


# In[15]:


array_slice[1]=12345


# In[16]:


array


# In[17]:


array_slice[:] = 64


# In[18]:


array


# In[23]:


array_2d = np.array([1,2,3,],[4,5,6],[7,8,9])


# In[25]:


array_2d = np.array([[1,2,3,],[4,5,6],[7,8,9]])


# In[26]:


array_2d


# In[27]:


array[7,8,9]


# In[28]:


array_2d([7,8,9])

array_2d[2]
# In[29]:


array_2d[2]


# In[30]:


array_2d[0][2]


# In[31]:


_3d = np.array([[1,2,3,],[4,5,6],[7,8,9],[10,11,12]])


# In[32]:


_3d


# In[33]:


_3d[0]


# In[34]:


_3d[0,0]


# In[35]:


_3d[1]


# In[36]:


_3d = np.array([[[1,2,3,],[4,5,6],[[7,8,9],[10,11,12]]])


# In[37]:


_3d


# In[38]:


_3d[0]


# In[39]:


_3d = np.array([[[1,2,3,],[4,5,6],[[7,8,9],[10,11,12]]])


# In[40]:


d3= np.array([[[1,2,3,],[4,5,6]],[[7,8,9],[10,11,12]]])


# In[41]:


d3[0]


# In[42]:


oldvals=d3[0].copy


# In[43]:


d3[0]


# In[44]:


d3[0]=42


# In[45]:


d3


# In[46]:


d3=oldvalues


# In[47]:


d3=oldvals


# In[48]:


d3


# In[49]:


d3


# In[ ]:





# In[50]:


d3


# In[51]:


print(d3)


# In[52]:


names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])


# In[53]:


data = np.random.randn(7,4)


# In[54]:


names


# In[55]:


data


# In[56]:


names == 'Bob'


# In[57]:


data[names=='Bob']


# In[58]:


arr = np.empty((8,4))


# In[59]:


for i in range(8):
    arr[i]=i


# In[60]:


arr


# In[61]:


arr[[-3,-5,-7]]


# In[62]:


arr=np.arange(32).reshape((8,4))


# In[63]:


arr


# In[65]:


x = np.random.randn(8)
      


# In[66]:


y = np.random.randn(8)
      


# In[67]:


x


# In[68]:


y


# In[69]:


np.maximum(x,y)


# In[1]:


import random
position=0
walk=[position]
steps=1000
for i in range(steps):
    step=1 if random.randint(0,1) else -1
    position+=step
    walk.append(position)


# In[2]:


plt.plot(walk[:100])


# In[3]:


plot(walk[:100])


# In[4]:


import numpy as np


# In[5]:


import plot as plt


# In[6]:


import matplotlib as plt


# In[7]:


plt.plot(walk[:100])


# In[8]:


import pandas as pd
from pandas import Series, DataFrame


# In[9]:


obj = pd.series([4,7,-5,3])


# In[10]:


obj = pd.Series([4,7,-5,3])


# In[11]:


obj


# In[12]:


obj,values


# In[13]:


obj.values


# In[14]:


obj.index


# In[15]:


obj2=pd.Series([4,7,-5,3], index=['d','b','a','c'])


# In[16]:


obj2


# In[17]:


obj2.index


# In[18]:


obj2['d']


# In[19]:


obj[obj2>3]


# In[20]:


obj2[obj2>3]


# In[21]:


'b' in obj2


# In[22]:


sdata = {'ohio':35000, 'texas':71000, 'oregeon':16000, 'utah':5000}


# In[23]:


obj3=pd.Series(sdata)


# In[24]:


obj3


# In[25]:


states = ['california','ohio','oregon','texas']


# In[26]:


obj4=pd.Series(sdata, index=states)


# In[27]:


obj4


# In[28]:


pd.isnull(obj4)


# In[29]:


pd.notnull(obj4)


# In[30]:


obj4.isnull()


# In[31]:


obj3


# In[32]:


obj3+obj4


# In[33]:


obj4.name = 'pop'


# In[34]:


obj4.index.name = 'state'


# In[35]:


obj4


# In[36]:


obj


# In[37]:


data = {}


# In[ ]:





# In[41]:


data = {
    'state':['OH', 'WV','NY'],
    'population':[ 100,200,300,],
    'people': ['bob','joe','karen']
    
    
    
}


# In[42]:


data


# In[43]:


pd.DataFrame(data)


# In[44]:


frame


# In[45]:


frame.head()


# In[46]:


frame = pd.DataFrame(data)


# In[47]:


frame


# In[48]:


frame.head()


# In[49]:


frame2= pd.DataFrame(data, columns=['month', 'year'], index= [1,2,3])


# In[50]:


frame2


# In[51]:


frame2= pd.DataFrame(frame, columns=['month', 'year'], index= [1,2,3])


# In[52]:


frame2


# In[53]:


frame2.columns()


# In[54]:


frame2.columns


# In[1]:


frame = pd.DataFrame(np.random.randn(4,3), columns=list'bde', index = ['Utah', 'Ohio', 'Texas','Oregon'] )


# In[2]:


frame = pd.DataFrame(np.random.randn(4,3), columns=list('bde'), index = ['Utah', 'Ohio', 'Texas','Oregon'] )


# In[3]:


import pandas as pd


# In[4]:


from pandas import DataFrame


# In[5]:


frame = pd.DataFrame(np.random.randn(4,3), columns=list('bde'), index = ['Utah', 'Ohio', 'Texas','Oregon'] )


# In[6]:


import numpy as np


# In[7]:


frame = pd.DataFrame(np.random.randn(4,3), columns=list('bde'), index = ['Utah', 'Ohio', 'Texas','Oregon'] )


# In[8]:


frame


# In[9]:


np.abs(frame)


# In[10]:


f = lambda x: x.max()- x.min()


# In[11]:


frame.apply(f)


# In[12]:


frame.apply(f, axis='columns')


# In[13]:


obj = pd.Series(range(4), index =['d','a','b','c'])


# In[14]:


obj


# In[15]:


obj.sort_index()


# In[16]:


frame = pd.DtaFrame(np.arrange8().reshape(2,4)), index=['three','one'], columbs=['d','a','b','c']


# In[17]:


frame = pd.DataFrame(np.arrange8().reshape(2,4)), index=['three','one'], columbs=['d','a','b','c']


# In[18]:


_frame = pd.DataFrame(np.arrange8().reshape(2,4)), index=['three','one'], columbs=['d','a','b','c']


# In[19]:


pd.DataFrame(np.arrange8().reshape(2,4)), index=['three','one'], columbs=['d','a','b','c']


# In[20]:


frame = pd.DataFrame(np.arrange8().reshape((2,4)), index=['three','one'], columbs=['d','a','b','c'])


# In[21]:


frame = pd.DataFrame(np.arrange(8).reshape((2,4)), index=['three','one'], columbs=['d','a','b','c'])


# In[22]:


frame = pd.DataFrame(np.arange(8).reshape((2,4)), index=['three','one'], columbs=['d','a','b','c'])


# In[23]:


frame = pd.DataFrame(np.arange(8).reshape((2,4)), index=['three','one'], columns=['d','a','b','c'])


# In[24]:


frame


# In[25]:


frame.sort_values(by='b')


# In[26]:


frame.sort_values(by=['a','b'])


# In[1]:


pd


# In[2]:


df = pd.DataFrame([[1.4,np.nan], [7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]], index=['a','b','c','d'], columns = ['one','two'])


# In[3]:


import pandas as pd
from pandas import DataFrame, Series


# In[4]:


df = pd.DataFrame([[1.4,np.nan], [7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]], index=['a','b','c','d'], columns = ['one','two'])


# In[5]:


import numpy as np.


# In[6]:


import numpy as np


# In[7]:


df = pd.DataFrame([[1.4,np.nan], [7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]], index=['a','b','c','d'], columns = ['one','two'])


# In[8]:


df


# In[9]:


df.sum()


# In[10]:


df.sum(axis='columns')


# In[11]:


df.idmax()


# In[12]:


df.idxmax()


# In[13]:


df.cumsum()


# In[14]:


df.describe()


# In[15]:


conda install pandas-datareader


# In[16]:


pip install pandas-datareader


# In[17]:


import pandas-datareader .data as web
all_data = {ticker: web.get_yahoo(ticker)}


# In[18]:


import pandas-datareader.data as web
all_data = {ticker: web.get_yahoo(ticker)
     for ticker in ['APPL', 'IBM', 'MSFT', 'GOOG']}


# In[19]:


import pandas_datareader.data as web
all_data = {ticker: web.get_yahoo(ticker)
     for ticker in ['APPL', 'IBM', 'MSFT', 'GOOG']}


# In[20]:


import pandas_datareader.data as web


# In[21]:


all_data = {ticker: web.get_data_yahoo(ticker)
     for ticker in ['APPL', 'IBM', 'MSFT', 'GOOG']}


# In[22]:


all_data = {ticker: web.get_data_yahoo(ticker)
     for ticker in ['APPL', 'IBM', 'MSFT', 'GOOG']}


# In[23]:


all_data = {ticker: web.get_data_yahoo(ticker)
     for ticker in ['APPL', 'IBM', 'MSFT', 'GOOG']}
price = pd.DataFrame({ticker: data ['Adj Close']
                     for ricker, data in all_data,items()})
volume = pd.DataFrame({ticker: data ['Volume']
                     for ricker, data in all_data,items()})


# In[24]:


all_data = {ticker: web.get_data_yahoo(ticker)
     for ticker in ['APPL', 'IBM', 'MSFT', 'GOOG']}
price = pd.DataFrame({ticker: data ['Adj Close']
                     for ricker, data in all_data.items()})
volume = pd.DataFrame({ticker: data ['Volume']
                     for ricker, data in all_data.items()})


# In[25]:


returns = price.pct_change()


# In[26]:


pip3 install pandas-datareader


# In[27]:


import datetime, quandl

ndq = quandl.get("NASDAQOMX/COMP-NASDAQ", 
              trim_start='2018-03-01', 
              trim_end='2018-04-03')

print(ndq.head(4))


# In[28]:


from pandas_datareader import data as pdr
import fix_yahoo_finance

data = pdr.get_data_yahoo('APPL', start='2017-04-23', end='2017-05-24')


# In[29]:


cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
data.reindex(columns=cols)


# In[30]:


import pandas_datareader.data as web
import datetime as dt

start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()
df = web.DataReader("TSLA", 'morningstar', start, end


# In[31]:


import pandas_datareader.data as web
import datetime as dt


# In[32]:


start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()
df = web.DataReader("TSLA", 'morningstar', start, end)


# In[33]:


aapl = data.DataReader("AAPL", 
                       start='2015-1-1', 
                       end='2015-12-31', 
                       data_source='yahoo')['Adj Close']


# In[34]:


aapl = web.DataReader("AAPL", 
                       start='2015-1-1', 
                       end='2015-12-31', 
                       data_source='yahoo')['Adj Close']


# In[35]:


aapl


# In[36]:


returns = aapl.pct_change()


# In[37]:


returns.tail()


# In[38]:


returns['Date'].corr(returns['Adj Close'])


# In[39]:


data = web.DataReader("AAPL", 
                       start='2015-1-1', 
                       end='2015-12-31', 
                       data_source='yahoo')['Adj Close']


# In[42]:


data__ = web.DataReader(("AAPL", 
                       start='2015-1-1', 
                       end='2015-12-31', 
                       data_source='yahoo')['Adj Close'],
                     ("IBM", 
                       start='2015-1-1', 
                       end='2015-12-31', 
                       data_source='yahoo')['Adj Close'])


# In[43]:


data = web.DataReader("AAPL", "IBM",
                       start='2015-1-1', 
                       end='2015-12-31', 
                       data_source='yahoo')['Adj Close']


# In[44]:


data = web.DataReader("AAPL", 
                       start='2015-1-1', 
                       end='2015-12-31', 
                       data_source='yahoo')['Adj Close']


# In[45]:


data1=web.DataReader("IBM", 
                       start='2015-1-1', 
                       end='2015-12-31', 
                       data_source='yahoo')['Adj Close']


# In[46]:


returns[data].corr(returns[data1])


# In[47]:


returns


# In[48]:


returns = data.pct_change()


# In[49]:


returns1 = data1.pct_change()


# In[50]:


returns.tail()


# In[51]:


returns1.tail()


# In[52]:


returns['AAPL'].corr(returns['IBM'])


# In[53]:


returns['data'].corr(returns['data1'])


# In[54]:


returns.corr(returns1)


# In[55]:


returns.cov(returns1)


# In[56]:


*100


# In[57]:


returns.cov()


# In[58]:


get_ipython().system('cat examples/ex1.csv')
a,b,c,d,message
1,2,3,4,hello


# In[1]:


import requests


# In[2]:


url = 'https://api.github.com/repos/pandas-dev/pandas/issues'


# In[3]:


respo = requests.get(url)


# In[4]:


respo


# In[5]:


data = respo.json()


# In[6]:


data


# In[7]:


data[0]['title']


# In[8]:


import pandas as pd
from pandas import DataFrame


# In[9]:


issues = pd.DataFrame(data,columns['number','title','labels','state'])


# In[10]:


issues = pd.DataFrame(data,columns =['number','title','labels','state'])


# In[11]:


issues


# In[12]:


import sqlite3


# In[13]:


query = ""
CREATE TABLE test
(a VARCHAR(20),
b VARCHAR(20),
 c REAL, d INTEGER);


# In[32]:


query = """CREATE TABLE test
(a VARCHAR(20),
b VARCHAR(20),
 c REAL, d INTEGER); """


# In[21]:


con = sqlite3.connect('mydata.sqlite')


# In[33]:


con.execute(query)


# In[34]:


con.commit()


# In[37]:


data = [('Atlanta', 'Georgia',  1.25, 6),('Tallahasee','Florida', 2.6,3),('Sacramento','California', 1.7,5)]


# In[38]:


stmt = "INSERT INTO test VALUES (?,?,?,?)"


# In[39]:


con.executemany(stmt,data)


# In[40]:


con.commit


# In[41]:


con.commit()


# In[42]:


cursor = con.execute('select * from test')


# In[43]:


cursor


# In[44]:


rows = cursor.fetchall()


# In[45]:


rows


# In[46]:


cursor.description


# In[47]:


pd.DataFrame(rows, columns=[x[0] for x in cursor.description])


# In[48]:


import sqlalchemy as sqla


# In[49]:


db = sqla.create_engine('sqlite:///mydata.sqlite')


# In[51]:


pd.read_sql('select * from test', db)


# In[53]:


import numpy as np
string_data= pd.Series(['aardvark', 'artichoke', np.nan, 'avacado'])

string_data
# In[54]:


string_data


# In[55]:


string_data.isnull()


# In[56]:


string_data[0] = None


# In[58]:


string_data.isnull()


# In[59]:


from numpy import nan as NA


# In[ ]:




