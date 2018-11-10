
# coding: utf-8

# # KrishiHub Data Science Intern Assignment?? (spelling nahi aata edit it)
# 
# You are required to do analyze the dataset and find out the following insights :-
# 
# 1. Top 10 keywords for funded projects and top 10 keywords for unfunded projects based
# on count.
# 
# 2. Top 10 fastest funded projects.
# 
# 3. Distribution of funded projects by country.
# 
# 4. The goal amount range where the projects have been funded the highest. (Feel free to
# use your assumption for goal ranges)
# 
# 5. How length and quality of project description correlates with project funding status.
# 
# 6. Top 10 successful projects sorted by backer count and top 10 unsuccessful projects
# sorted by backer count.
# 
# 7. How number of backers correlates with project success.
# 
# 8. Successful and unsuccessful projects histogram (by count) per category (you will need
# to figure out the category of the project from the description, example - film, music, art
# etc.)

# ***
# ## 1.0 Loading the Dataset
# 
# 

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('./data/kickstarter.csv')


# In[3]:


data.head()


# ***
# ## 2.0 Top 10 keywords for funded projects and top 10 keywords for unfunded projects
# 

# In[4]:


# importing the needed stuffs
from collections import Counter


# We'll create three Counter objects, one for keywords from funded projects, one for keywords from unfunded projects, and one for all the keywords.

# In[5]:


funded_counts = Counter()
unfunded_counts = Counter()
total_counts = Counter()


# In[6]:


data.keywords[0].split("-")


# In[7]:


# Loop over all the keywords in all the data.keywords and increment the counts in the appropriate counter objects
for i in range(len(data.keywords)):
    if(data.final_status[i] == 1):
        for kword in data.keywords[i].split("-"):
            funded_counts[kword] += 1
            total_counts[kword] += 1
    else:
        for kword in data.keywords[i].split("-"):
            unfunded_counts[kword] += 1
            total_counts[kword] += 1


# Awesome now we have the Dicts mapped up, now we Need to examine the Dicts

# In[ ]:


# Examining the Dicts for most common keywords in funded projects
funded_counts.most_common()


# In[ ]:


# Examining the Dicts for most common keywords in unfunded projects
unfunded_counts.most_common()


# Keywords like 'the', 'a', 'for' are used more often in both funded and unfunded keyword counts, That's creating Noise and doesn't not convey anything.(except that we use pronouns and articles alot :P)

# We can tackle this problem with a 3rd grade math tool : "Ratios" 

# In[10]:


def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def fund2unfund_ratio(counts, f_counts,uf_counts, threshold):
    fund2unfund_ratios = Counter()

    # We can calculate the ratios of funded(a) to unfunded(b) uses of the most common keywords, as  (A / (B+1))
    # We need to consider a threshold for keywords to be "common" or uncommon, let's say if the keyword have been used more than 100 times it's uncommon
    for term,cnt in list(counts.most_common()):
        if(cnt > threshold):
            fund2unfund_r = (float(f_counts[term]) - float(uf_counts[term]))/ float(counts[term])
            fund2unfund_ratios[term] = sigmoid(fund2unfund_r) # we Return Sigmoid to avoid negetive numbers
    return fund2unfund_ratios


# In[11]:


# let's examine if the the thing works out or not
fund2unfund_ratios = fund2unfund_ratio(total_counts, funded_counts, unfunded_counts, 100)
print("Funded to Unfunded ratio for 'show' = {}".format(fund2unfund_ratios["show"]))
print("Funded to Unfunded ratio for 'me' = {}".format(fund2unfund_ratios["me"]))
print("Funded to Unfunded ratio for 'documentary' = {}".format(fund2unfund_ratios["documentary"]))
print("Funded to Unfunded ratio for 'vol' = {}".format(fund2unfund_ratios["vol"]))


# In[ ]:


fund2unfund_ratios.most_common()


# Looking closely at the values you just calculated, we see the following:
# 
# 1. Keywords that we would see more often in funded projects – like "docunmentary" – have a ratios greater than 0.5. 
# 2. Keywords that we would see more often in unfunded projects – like "show" – have ratios that are less than 0.5.
# 3. Other normal keywords, which don't really convey anything because one would expect to see them in all sorts of keyword mapping is scattered around 0.5
# 
# A perfectly neutral keyword – one that was used in exactly the same number of funded keywords as unfunded keywords – should be almost exactly 0 as the math says. We dont see that much of that in the data, which says that there is a possibility of noise in the data that we need to spread out.
# 
# And the tool we are going to use now is "Set Theory"

# In[ ]:


common_counts = Counter()

common_counts = funded_counts & unfunded_counts
common_counts.most_common()


# In[ ]:


rfk = funded_counts - common_counts
rfk.most_common()


# In[15]:


rufk = unfunded_counts - common_counts
rf2f_ratios = fund2unfund_ratio(total_counts, rfk, rufk, 100)


# In[ ]:


rf2f_ratios.most_common()


# Now we can clearly see the data properly spread out between 1 and 0 with much less noise

# In[17]:


reverse_rf2f_ratios = {l:k for k,l in sorted([(j,i) for i,j in rf2f_ratios.items()])}


# In[18]:


from itertools import islice

def top10unfunded(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


# ***
# ### 2.1 Top 10 keywords for funded projects
# 

# In[19]:


for word, val in rf2f_ratios.most_common(10):
    print("{}".format(word) )


# ***
# ### 2.2 Top 10 keywords for unfunded projects
# 

# In[20]:


for word, val in (top10unfunded(10,reverse_rf2f_ratios.items())):
    print("{}".format(word) )


# ******
# 
# ## 3.0 Top 10 fastest funded projects

# In[21]:



"""
cols = ['project_id', 'name', 'time_took']
ffp = pd.DataFrame(columns=cols)
for i in range(len(data)):
    if (data.final_status[i] == 1):
        time = float(data.state_changed_at[i]) - float(data.launched_at[i])
        ffp = ffp.append(pd.Series([data.project_id[i], data.name[i], time], index=cols),ignore_index=True)
ffp.to_csv('fastestfundedproject.csv', ind)
"""


# In[22]:


ffp=pd.read_csv('fastestfundedproject.csv')


# ### 3.1 Top 10 Fastest Funded Project

# In[23]:


ffp10 = ffp.sort_values(by=['time_took']).head(10)


# In[24]:


ffp10


# *****
# ## 4.0 Distribution of funded projects by country

# In[25]:


country_distrib = Counter()


# In[26]:


for i in range(len(data)):
    if (data.final_status[i]==1):
        country_distrib[data.country[i]] += 1


# In[27]:


countryD = dict(country_distrib)
country_distribution = pd.DataFrame(dict(country = list(countryD.keys()),
                                         project_count = list(countryD.values()))
                                   )


# In[28]:


country_distribution


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[30]:


a = sns.catplot(x="country", y="project_count",kind="bar",palette="Blues_d",data=country_distribution)


# We see that most of the Projects are frm the US

# ***

# ## 5.0 The goal amount range where the projects have been funded the highest.

# In[31]:


funded = data[data['final_status']>0]
goalsorted = funded.sort_values(by=['goal'], ascending=False)
goaltop1per = goalsorted.head(int(len(goalsorted)*.01))


# We are taking the top 1% of the funded projects with highest goals in decending order to find the goal range

# In[32]:


print("The Goal amount of highest funded project ranges from :{} - {} ".format(float(((goaltop1per.head(1))['goal'])),float(((goaltop1per.tail(1))['goal']))))


# ***
# ## 6.0 How length and quality of project description correlates with project funding status

# In[33]:


# we can get the project length by simply computing the time between deadline and creation date
data['project_length'] = data.deadline - data.created_at


# 
# Getting the quality of a project's description can be done by many machine learning techniques but for now let's rate the description of the project  via it's sentimental ratings
# 

# In[74]:


from textblob import TextBlob


# TextBlob is a Python library for processing textual data. It provides functions for common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, and more. 
# 
# Here we shall judge the quality of Description of project by the subjectivity of the project description.

# In[114]:


quality = []


# In[115]:




for text in data.desc:
    blob = TextBlob(str(text))
    quality.append(blob.sentiment.subjectivity)


# In[116]:


len(quality)


# In[118]:


data['desc_quality'] = quality


# In[120]:


data.head()


# In[121]:


correlation= data.corr()


# In[123]:


correlation['project_length']['desc_quality']


# No correlation as per our current hypothesis

# ***
# ## 7.0 Top 10 successful projects sorted by backer count and top 10 unsuccessful projects sorted by backer count.

# In[34]:


unfunded = data[data['final_status'] == 0]


# In[35]:


backersortedfunded = funded.sort_values(by=['backers_count'], ascending=False)
backersortedunfunded = unfunded.sort_values(by=['backers_count'], ascending=False)


# ### 7.1 Top 10 successfull projects sorted by backer count(high-low)

# In[36]:


backersortedfunded.head(10)


# ### 7.1 Top 10 unsuccessfull projects sorted by backer count(high-low)

# In[37]:


backersortedunfunded.head(10)


# ***
# ## 8.0 How number of backers correlates with project success.

# In[38]:


# We need to normalize backers count
average = data["backers_count"].mean()
SD = data["backers_count"].std()

data["norm_backers_count"] = sigmoid((data.backers_count -average)/SD)


# In[39]:


correlation= data.corr()


# In[40]:


correlation['final_status']['norm_backers_count']


# We see that final status of the project slightly positively correalates with the norm backers count.

# ***
# ### 9.0 Successful and unsuccessful projects histogram (by count) per category

# In[41]:


# Loop over all the keywords in all the data.keywords and increment the counts in the appropriate counter objects
from textblob import TextBlob
blob = TextBlob(data.keywords[6])


# In[42]:


data.keywords[6]


# In[43]:


blob = blob.replace("-"," ")


# In[44]:


blob.noun_phrases


# In[45]:


blob.sentiment


# In[46]:


def get_category(txt):
    category_counte= Counter()

    a=[]
    blob = TextBlob(str(txt))
    blob = blob.replace("-"," ")
    for item in list(blob.noun_phrases):
        bob = TextBlob(item)
        category_counte[item] = sigmoid((float(bob.polarity))/((float(bob.subjectivity))+1))
    
    if (len(list(category_counte.most_common()))>0):
        a = list((category_counte.most_common())[0])
        return str(a[0])
    else:
        return "others"


# In[47]:


categ=[]


# In[48]:


for i in range(len(data.keywords)):
    categ.append(get_category(data["keywords"][i]))


# 
# 

# In[ ]:


categ


# In[50]:


data["test"] = categ


# In[51]:


data.head(7)


# In[52]:


countercategorized = Counter()


# In[53]:


for sentances in categ:
    
    sentance = sentances.split(" ")
    for words in sentance:
        countercategorized[words] += 1
    


# In[54]:


top30categ=list(dict(countercategorized.most_common(30)))


# In[55]:


top30categ


# In[56]:


category30list=[]
for sentances in categ:
    sentance = sentances.split(" ")
    for words in sentance:
        if (words in top30categ):
            category30list.append(words) 
            break
        else:
            category30list.append('others')
            break
            
    


# In[57]:


data["category"] = category30list


# In[58]:


data.head()


# In[59]:


unfunded = data[data['final_status'] == 0]
funded = data[data['final_status'] == 1]


# In[60]:


funded_counter = Counter()
unfunded_counter = Counter()


# In[61]:


for word in funded.category:
    if (word != 'others'):
        funded_counter[word] += 1
for word in unfunded.category:
    if (word != 'others'):
        unfunded_counter[word] += 1


# In[62]:


categories = (list(funded_counter.keys()))
fundedcount = (list(funded_counter.values()))
unfundedcount = (list(unfunded_counter.values()))


# In[63]:


categorydatas ={}

categorydatas = { 'categories' : categories, 'funded_count' : fundedcount, 'unfunded_count' : unfundedcount}
categorydata = pd.DataFrame.from_dict(categorydatas)


# In[64]:


sns.set(font_scale=1.2)


# ### 9.1 Unsuccessfull Project Count per category (almost)

# In[73]:


histogrm= sns.catplot(x="categories", y="funded_count",kind="bar",palette="Blues_d",data=categorydata, height=8, aspect=29/8)


# ### 9.2 Unsuccessfull Project Count per category (almost)

# In[72]:


histogrm= sns.catplot(x="categories", y="unfunded_count",kind="bar",palette="Blues_d",data=categorydata, height=8, aspect=29/8  )

