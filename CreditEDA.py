#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# ignoring warnings

import warnings
warnings.simplefilter('ignore')


# In[3]:


# reading the application file

app = pd.read_csv('application_data.csv')
app.head()


# In[4]:


# reading previous application data file

prev_app = pd.read_csv('previous_application.csv')
prev_app.head()


# In[5]:


# display shape
app.shape


# In[6]:


prev_app.shape


# In[7]:


# describings columns in application dataset
app.describe()


# In[8]:


prev_app.describe()


# In[9]:


# display columns in dataset

app.columns


# In[10]:


prev_app.columns


# #### Data cleaning in application dataset

# In[11]:


# checkings null values
null_df=app.isnull().sum()
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
null_df


# In[12]:


percent_df= round((100*app.isnull().sum()/len(app)),2)
percent_df


# In[13]:


# removing all null values more than 20%
app=app.loc[:,app.isnull().mean()<=.20]


# In[14]:


round((100*app.isnull().sum()/len(app)),2)


# In[15]:


len(app.columns)


# In[ ]:





# #### Data cleaning previous application dataset

# In[16]:


# checkings null values
null_df_prev=prev_app.isnull().sum()
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
null_df_prev


# In[17]:


percent_df_prev= round((100*prev_app.isnull().sum()/len(prev_app)),2)
percent_df_prev


# In[18]:


# removing all null values more than 20%
prev_app=prev_app.loc[:,prev_app.isnull().mean()<=.19]


# In[19]:


round((100*prev_app.isnull().sum()/len(prev_app)),2)


# In[20]:


len(prev_app.columns)


# now seems no such values in prev application data

# ### Imputing missing values

# imputing missing values in Name_Type_Suite

# In[21]:


app.NAME_TYPE_SUITE.value_counts()


# In[22]:


app.NAME_TYPE_SUITE.mode()


# In[23]:


# filling NMAE_TYPE_SUITE with most occured value

app.NAME_TYPE_SUITE.fillna(value="Unaccompanied", inplace=True)


# In[24]:


# checking mean value of AMT_REQ_CREDIT_BUREAU_HOUR

app.AMT_REQ_CREDIT_BUREAU_HOUR.mean()


# In[25]:


# filling 'AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_YEAR' with 0, since mean is near to 0

app.loc[:,'AMT_REQ_CREDIT_BUREAU_HOUR':'AMT_REQ_CREDIT_BUREAU_YEAR']=app.loc[:,'AMT_REQ_CREDIT_BUREAU_HOUR':'AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(value=0)


# In[26]:


# checking mean of AMT_GOODS_PRICE

app.AMT_GOODS_PRICE.mean()


# In[27]:


# filling AMT_GOODS_PRICE with mean value

app.AMT_GOODS_PRICE.fillna(value=app.AMT_GOODS_PRICE.mean(),inplace=True)


# In[28]:


# checking AMT_ANNUITY

app.AMT_ANNUITY.plot.box()


# In[29]:


# since there outlier in AMT_ANNUITY, so filling with median

app.AMT_ANNUITY.fillna(value=app.AMT_ANNUITY.median(),inplace=True)


# In[30]:


round((100*app.isnull().sum()/len(app)),2)


# checking datatypes of columns

# In[31]:


app.info()


# In[32]:


# converting data type of some columns which need not to be float

app.DAYS_REGISTRATION = app.DAYS_REGISTRATION.astype(int, errors='ignore')
app.CNT_FAM_MEMBERS = app.CNT_FAM_MEMBERS.astype(int, errors='ignore')
app.OBS_30_CNT_SOCIAL_CIRCLE = app.OBS_30_CNT_SOCIAL_CIRCLE.astype(int, errors='ignore')
app.DEF_30_CNT_SOCIAL_CIRCLE = app.DEF_30_CNT_SOCIAL_CIRCLE.astype(int, errors='ignore')
app.OBS_60_CNT_SOCIAL_CIRCLE = app.OBS_60_CNT_SOCIAL_CIRCLE.astype(int, errors='ignore')
app.DEF_60_CNT_SOCIAL_CIRCLE = app.DEF_60_CNT_SOCIAL_CIRCLE.astype(int, errors='ignore')
app.AMT_REQ_CREDIT_BUREAU_HOUR = app.AMT_REQ_CREDIT_BUREAU_HOUR.astype(int, errors='ignore')
app.AMT_REQ_CREDIT_BUREAU_DAY = app.AMT_REQ_CREDIT_BUREAU_DAY.astype(int, errors='ignore')
app.AMT_REQ_CREDIT_BUREAU_WEEK = app.AMT_REQ_CREDIT_BUREAU_WEEK.astype(int, errors='ignore')
app.AMT_REQ_CREDIT_BUREAU_MON = app.AMT_REQ_CREDIT_BUREAU_MON.astype(int, errors='ignore')
app.AMT_REQ_CREDIT_BUREAU_QRT = app.AMT_REQ_CREDIT_BUREAU_QRT.astype(int, errors='ignore')
app.AMT_REQ_CREDIT_BUREAU_YEAR = app.AMT_REQ_CREDIT_BUREAU_YEAR.astype(int, errors='ignore')


# In[33]:


# remove unwanted columns

unwanted_columns=['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL',
                  'REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','DAYS_LAST_PHONE_CHANGE','FLAG_DOCUMENT_2',
                 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8',
                 'FLAG_DOCUMENT_21','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
                 'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
                 'FLAG_DOCUMENT_20']

app.drop(labels=unwanted_columns,axis=1,inplace=True)


# In[34]:


# checking Gender column
app.CODE_GENDER.value_counts()


# In[35]:


app.CODE_GENDER.replace(to_replace='XNA', value='F', inplace=True)


# In[36]:


# checking Organization column
app.ORGANIZATION_TYPE.value_counts()


# In[37]:


# dropping rows having XNA value in organization, because total 55374 values are XNA

app = app.drop(app.loc[app.ORGANIZATION_TYPE=='XNA'].index)


# In[38]:


# converting AMT_INCOME_TOTAL to categorical values

bins_income = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000,app.AMT_INCOME_TOTAL.max()]
slot_income = ['0-25000','25000-50000','50000-75000','75000-100000',
               '100000-125000','125000-150000','150000-175000','175000-200000',
               '200000-225000','225000-250000','250000-275000','275000-300000',
               '300000-325000','325000-350000','350000-375000','375000-400000',
               '400000-425000','425000-450000','450000-475000','475000-500000', '500000 and above']

app["AMT_INCOME_RANGE"]=pd.cut(app.AMT_INCOME_TOTAL,bins_income,labels=slot_income)


# In[39]:


# converting AMT_CREDIT to categorical values

bins_credit = [0,150000,200000,250000,300000,350000,400000,450000,500000,550000,650000,700000,750000,800000,850000,900000,app.AMT_CREDIT.max()]
slot_credit = ['0-150000','150000-200000','200000-250000','250000-300000','300000-350000',
               '350000-400000','400000-450000','450000-500000','500000-550000','550000-600000','600000-650000','650000-700000',
               '700000-800000','800000-850000','850000-900000','900000 and above']

app["AMT_CREDIT_RANGE"]=pd.cut(app.AMT_CREDIT,bins_credit,labels=slot_credit)


# In[40]:


# checking TARGET_DATA

app.TARGET.value_counts()


# In[41]:


app.TARGET.value_counts().plot.barh(color='Green')
plt.title("Target")
plt.xlabel("Count")
plt.ylabel("Target")
plt.show()


# ### Analysis

# In[42]:


# Plotting income range across various gender

target_0=app.loc[app.TARGET==0]
target_1=app.loc[app.TARGET==1]
sns.countplot(data=target_0, x="AMT_INCOME_RANGE", hue="CODE_GENDER", palette="Blues")
plt.xticks(rotation=90)
plt.title("Distribution of income range")
plt.xlabel("Income Range")
plt.ylabel("Count")
plt.show()


# Conclusion from graph:
# - Income range from 125000 to 150000 is having more number of credits
# - Very less count from range 450000-475000
# - It seems that the females are more than male in having credit range: 125000 - 150000

# In[43]:


# plotting for the various Income types accross various Gender.

sns.countplot(data=target_0, x="NAME_INCOME_TYPE", hue="CODE_GENDER", palette='copper')
plt.xticks(rotation=60)
plt.title("Distribution of Income Type")
plt.xlabel("Income Type")
plt.ylabel("Count")
plt.show()


# Conclusion from the Graph:
# - working women has more credit than others
# - "State servant","Working" and "Commercial associate" have more credit counts compared to others

# In[44]:


# Plotting contract type accross various gender

sns.countplot(data=target_0, x="NAME_CONTRACT_TYPE", hue="CODE_GENDER", palette='copper')
plt.xticks(rotation=60)
plt.title("Distribution of Income Type")
plt.xlabel("Contract Type")
plt.ylabel("Count")
plt.show()


# Conclusion from graph
# - Cash loans is having more credits than "Revolving loans" contract type
# - Female applies more for credit

# In[45]:


# Plotting for Organization type
plt.figure(figsize=[15,30])
sns.countplot(data=target_0, y="ORGANIZATION_TYPE",order=target_0["ORGANIZATION_TYPE"].value_counts().index, palette='cool')
plt.xticks(rotation=60)
plt.title("Distribution of Organization Type")
plt.xlabel("Organization Type")
plt.ylabel("Count")
plt.show()


# Conclusion from graph
# - "Business Entity Type 3", "Self-employed","Other" and "Medicine" are organization type which have applied for credits
# - less are from "Industry type 8",type 13, "Trade type 5", type 4 and religion

# In[46]:


# Plotting Income range accross gender for Target=1

sns.countplot(data=target_1, x="AMT_INCOME_RANGE", hue="CODE_GENDER", palette="Blues")
plt.xticks(rotation=90)
plt.title("Distribution of income range")
plt.xlabel("Income Range")
plt.ylabel("Count")
plt.show()


# Conclusion from Graph
# - Male counts are higher
# - Income range from 100000 to 200000 is having more number of credits
# - less for 450000-475000

# In[47]:


# plotting for the various Income types accross various Gender for TARGET=1

sns.countplot(data=target_1, x="NAME_INCOME_TYPE", hue="CODE_GENDER", palette='copper')
plt.xticks(rotation=60)
plt.title("Distribution of Income Type")
plt.xlabel("Income Type")
plt.ylabel("Count")
plt.show()


# Conclusion from Graph
# - "working", "commmercial associate", "state servant" income type has higher credits
# - female have more credits

# In[48]:


# Plotting contract type accross various gender for TARGET=1

sns.countplot(data=target_1, x="NAME_CONTRACT_TYPE", hue="CODE_GENDER", palette='copper')
plt.xticks(rotation=60)
plt.title("Distribution of Income Type")
plt.xlabel("Contract Type")
plt.ylabel("Count")
plt.show()


# Conclusio from Graph
# - "cash loans" has higher credits
# - female have more credits

# In[49]:


# Plotting for Organization type for TARGET=1
plt.figure(figsize=[15,30])
sns.countplot(data=target_1, y="ORGANIZATION_TYPE",order=target_0["ORGANIZATION_TYPE"].value_counts().index, palette='cool')
plt.xticks(rotation=60)
plt.title("Distribution of Organization Type")
plt.xlabel("Organization Type")
plt.ylabel("Count")
plt.show()


# Conclusion from Graph
# - "Business Entity Type 3", "Self-employed","Other" and "Medicine" are organization type which have applied for credits
# - less are from "Industry type 8",type 13, "Trade type 5", type 4 and religion

# ### Correlation

# In[50]:


# correlation among the target_0 people

target_0_corr = target_0.iloc[0:,2:].corr()
target_0_corr


# In[51]:


# plotting the correlation for the target_0

plt.figure(figsize=[14,9])
sns.heatmap(target_0_corr, annot=False, cmap='RdYlGn')
plt.title("Correlation for Target=0")
plt.show()


# Conclusion from graph
# - credit amount is inversely propotional to date of birth
# - credit amount is inversely propotional to number of children cilent have
# - income amount is inversely propotional to number of children cilent have
# - less children client have in densely populated area
# - credit amount is higher in densely populated area

# In[52]:


# correlation among the target_1 people

target_1_corr = target_1.iloc[0:,2:].corr()
target_1_corr


# In[53]:


# plotting the correlation for the target_1

plt.figure(figsize=[14,9])
sns.heatmap(target_1_corr, annot=False, cmap='RdYlGn')
plt.title("Correlation for Target=1")
plt.show()


# Conclusion from graph
# - client's permanent address does not match contact address are having less children 
# - client's permanent address does not match work address are having less children 
# - credit amount is inversely propotional to date of birth
# - credit amount is inversely propotional to number of children cilent have
# - income amount is inversely propotional to number of children cilent have
# - less children client have in densely populated area
# - credit amount is higher in densely populated area

# In[54]:


# finding top 10 correlations for target 0 and target 1
# converting the negative values to positive values and sorting values for target 0

corr_0 = target_0_corr.abs().unstack().sort_values(kind='quicksort').dropna()
corr_0 = corr_0[corr_0 != 1.0]
corr_0


# In[55]:


# top 10 correlation for target 0
corr_0.tail(10)


# In[56]:


# converting the negative values to positive values and sorting values for target 1

corr_1 = target_1_corr.abs().unstack().sort_values(kind='quicksort').dropna()
corr_1 = corr_1[corr_1 != 1.0]
corr_1


# In[57]:


# top 10 correlation for target 1
corr_1.tail(10)


# ### Bivariate Analysis of the numerical columns

# In[58]:


# plotting scatterplot to find any correlation and to check the trends in the dataset

plt.figure(figsize=[16,8])

plt.subplot(1,2,1)
sns.scatterplot(target_0.AMT_CREDIT, target_0.AMT_INCOME_TOTAL)
plt.title("Income vs Credit for target 0")
plt.yscale("log")
plt.xlabel("Credit")
plt.ylabel("Income")

plt.subplot(1,2,2)
sns.scatterplot(target_1.AMT_CREDIT, target_1.AMT_INCOME_TOTAL)
plt.title("Income vs Credit for target 1")
plt.yscale("log")
plt.xlabel("Credit")
plt.ylabel("Income")


# In[59]:


# plotting scatterplot to find any correlation and to check the trends in the dataset

plt.figure(figsize=[16,8])

plt.subplot(1,2,1)
sns.scatterplot(target_0.AMT_CREDIT, target_0.AMT_GOODS_PRICE)
plt.title("Goods price vs Credit for target 0")
plt.yscale("log")
plt.xlabel("Credit")
plt.ylabel("Goods Price")

plt.subplot(1,2,2)
sns.scatterplot(target_1.AMT_CREDIT, target_1.AMT_GOODS_PRICE)
plt.title("Goods price vs Credit for target 1")
plt.yscale("log")
plt.xlabel("Credit")
plt.ylabel("Goods Price")


# Conclusion from graph
# - AMT_CREDIT and AMT_GOODS_PRICE are highly correlated which means if increase in goods price, credit increased directly

# ### Analysis

# In[60]:


# Distribution of income amount for target=0
sns.boxplot(data=target_0, y='AMT_INCOME_TOTAL')


# Conclusion from graph
# - there are outliers present
# - also there is equal distribution of income amount of clients

# In[61]:


# Distribution of credit for target=0
sns.boxplot(data=target_0, y='AMT_CREDIT')


# Conclusion from graph
# - client credit lies in first quartile
# - there are outlier present

# In[62]:


# Distribution of ANNUITY for target=0
sns.boxplot(data=target_0, y='AMT_ANNUITY')


# Conclusion from graph
# 
# - client annuity lies in first quartile
# - there are outlier present

# In[63]:


# Distribution of income amount for target=1
sns.boxplot(data=target_1, y='AMT_INCOME_TOTAL')


# Conclusion from graph
# 
# - client income lies in third quartile
# - there are outlier present

# In[64]:


# Distribution of credit for target=1
sns.boxplot(data=target_1, y='AMT_CREDIT')


# Conclusion from graph
# 
# - client credit lies in first quartile
# - there are outlier present

# In[65]:


# Distribution of ANNUITY for target=1
sns.boxplot(data=target_1, y='AMT_ANNUITY')


# ### Multivariate Analysis

# In[67]:


# box plotting for target=0, credit amount

plt.figure(figsize=[16,12])
sns.boxplot(data=target_0, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT',hue='NAME_FAMILY_STATUS',orient='v', palette='Set1')
plt.xlabel("Education Type")
plt.ylabel("credit amount")
plt.title("Credit amount vs Education status Target=0")
plt.show()


# Conclusion from graph
# - family status of 'civil marriage','marriage' and 'separated' of academic degree education are having higher number of credits than others. also,higher education of family status 'marriage', 'single' and 'civil marriage' are having more outliers. Civil marriage for academic degree is having most of the credits in third quartiel

# In[68]:


# box plotting for target=0, income amount

plt.figure(figsize=[16,12])
sns.boxplot(data=target_0, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL',hue='NAME_FAMILY_STATUS',orient='v', palette='Set1')
plt.xlabel("Education Type")
plt.ylabel("Income Amount")
plt.title("Income amount vs Education status Target=0")
plt.show()


# Conclusion from graph
# - Education Type 'Higher Education' the income amount is mostly equal with family status. It does contain many outliers. Less outliers are having for Academic degree but there income amount is little higher that Higher education. Lower secondary of civil marriage family status are have less income amount than others.

# In[69]:


# box plotting for target=1, credit amount

plt.figure(figsize=[16,12])
sns.boxplot(data=target_1, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT',hue='NAME_FAMILY_STATUS',orient='v', palette='Set1')
plt.xlabel("Education Type")
plt.ylabel("credit amount")
plt.title("Credit amount vs Education status Target=1")
plt.show()


# Conclusion from graph
# - family status of 'civil marriage','marriage' and 'separated' of academic degree education are having higher number of credits than others. Most of the outliers are from Education type 'Higher Education' and 'Secondary'. Civil marriage for academic degree is having most of the credits in third quartiel

# In[70]:


# box plotting for target=1, income amount

plt.figure(figsize=[16,12])
sns.boxplot(data=target_1, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL',hue='NAME_FAMILY_STATUS',orient='v', palette='Set1')
plt.xlabel("Education Type")
plt.ylabel("Income Amount")
plt.title("Income amount vs Education status Target=1")
plt.show()


# ## Previous application dataset

# In[71]:


prev_app.head()


# In[72]:


# checking the NAME_CASH_LOAN_PURPOSE column for unique values

prev_app.NAME_CASH_LOAN_PURPOSE.value_counts()


# In[73]:


# removing 'XNA' and 'XAP' column values from the columns

prev_app = prev_app.drop(prev_app[prev_app.NAME_CASH_LOAN_PURPOSE=='XNA'].index)
prev_app = prev_app.drop(prev_app[prev_app.NAME_CASH_LOAN_PURPOSE=='XAP'].index)


# ## Merge two datasets, i.e. application dataset and previous application dataset

# In[74]:


# merging of the two datasets

merge_data=pd.merge(left=app,right=prev_app, how="inner", on="SK_ID_CURR",suffixes="_x")
merge_data.head()


# In[75]:


merge_data.columns


# In[82]:


# renaming the colmmns in the merge_data datasets

merge_data=merge_data.rename({'NAME_CONTRACT_TYPE_':'NAME_CONTRACT_TYPE','AMT_CREDIT_':'AMT_CREDIT',
                              'WEEKDAY_APPR_PROCESS_START_':'WEEKDAY_APPR_PROCESS_START',
                              'HOUR_APPR_PROCESS_START_':'HOUR_APPR_PROCESS_START',
                              'NAME_CONTRACT_TYPEx':'NAME_CONTRACT_TYPE_PREV','AMT_CREDITx':'AMT_CREDIT_PREV',
                              'WEEKDAY_APPR_PROCESS_STARTx':'WEEKDAY_APPR_PROCESS_START_PREV',
                              'HOUR_APPR_PROCESS_STARTx':'HOUR_APPR_PROCESS_START_PREV'},axis=1)


# In[84]:


# removing unwanted columns from merge_data

unwanted=['SK_ID_CURR','WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION',
          'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY',
          'REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY','WEEKDAY_APPR_PROCESS_START_PREV',
          'HOUR_APPR_PROCESS_START_PREV','FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY']

merge_data.drop(unwanted,axis=1,inplace=True)


# ### Analysis

# In[85]:


# plotting for contract status

plt.figure(figsize=[15,28])
sns.countplot(data=merge_data, y="NAME_CASH_LOAN_PURPOSE",
              order=merge_data.NAME_CASH_LOAN_PURPOSE.value_counts().index,hue="NAME_CONTRACT_STATUS",palette="magma")
plt.title("Distribution of contract status with purpose")
plt.xlabel("Count")
plt.ylabel("Loan Purpose")
plt.show()


# Conclusion from graph
# - loan purposes with 'Repairs' are facing more difficulties in payment on time
# - there are few places where loan payment is signficant higher than facing difficulties. They are 'Buying a Garage','Business development','Buying land','Buying a new car', and 'Education' Hence we can focus on these purposes for which the client is having for minimal payment difficulties

# In[88]:


# plotting for credit amount to logarithmic scale

plt.figure(figsize=[20,12])
sns.barplot(data=merge_data,x="NAME_CASH_LOAN_PURPOSE", hue="NAME_INCOME_TYPE",y="AMT_CREDIT_PREV",orient='v',palette="BuPu")
plt.xticks(rotation=90)
plt.ylabel("Amount Credit Prev")
plt.xlabel("Loan Purpose")
plt.title("Prev Credit amount vs Loan Purpose")
plt.show()


# Conclusion from graph
# - credit amount of loan purposes like "Buying a land","Buying a home","Buying a new car" and Building a house is higher
# - Income type of state servants have a significant amount of credit applied
# - Money for third person or a Hobby is having less credits applied for

# In[89]:


# plotting for credit amount to logarithmic scale

plt.figure(figsize=[20,12])
sns.barplot(data=merge_data,x="NAME_HOUSING_TYPE", hue="TARGET",y="AMT_CREDIT_PREV",orient='v',palette="autumn")
plt.xticks(rotation=90)
plt.ylabel("Amount Credit Prev")
plt.xlabel("Housing Types")
plt.title("Prev Credit amount vs Housing Type")
plt.show()


# Conclusion from graph
# - office apartment is having higher credit of target 0 and co-op apartment is having higher credit of target=1.
# - bank should avoid giving loans to the housing type of co-op apartment as they are having difficulties in payment.
# - bank can focus  mostly on housing type with parents or House\appartment or muncipal apartment for sucessful payment.

# ## Conclusion

# - banks should approve loans more for office apartment, co-op apartment housing type as there are less payment difficulties
# - banks should provide loans to 'Repairs' & 'Others' purposes.
# - banks should provide loans to the 'Business Entity Type-3' and 'Self-Employed' persons.

# In[ ]:




