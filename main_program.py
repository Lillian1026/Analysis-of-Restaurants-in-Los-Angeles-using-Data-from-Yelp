#!/usr/bin/env python3
# coding: utf-8

from bs4 import BeautifulSoup
from pandas import read_csv
import requests
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.formula.api as ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from langdetect import detect_langs
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from sklearn.feature_extraction.text import CountVectorizer

url='https://api.yelp.com/v3/businesses/search'
key='tdqUcll2xg2Gwzj1W7sUsYfPQm2Cbdcn1QyJXsTfN1bWpn--D2cJgg-enps5Baa_0hVxyxDSz7scVH4pygP_3DffkHhn-9rUN8e1Ajsyt9Rh67Wtv-iABuusRrJTYnYx'
headers={'Authorization': 'Bearer %s' %key}

offset_number=[0,50,100,150,200,250,300,350,400,450]
response_all=[]
for i in offset_number:
    parameters={'location': 'Los Angeles, CA 90089',
           'term':'restaurant',
           'radius':5000,
           "offset": i,
           "limit": 50}
    response=requests.get(url,headers=headers,params=parameters)
    r=response.json()
    response_all.append(r) 

def to_pd_df(query):
    results_usc={'Name':[],'Category':[],'Rating':[],'Price':[],'Review_count':[],'Location':[],'Coordinates':[],'URL':[],'business_id':[]}
    for q in query:
        results_usc['Name'].append(q['name'])
        results_usc['Category'].append(q['categories'][0]['alias'])
        results_usc['Rating'].append(q['rating'])
        try:
            results_usc['Price'].append(int(len(q['price'])))
        except:
            results_usc['Price'].append(None)
        results_usc['Review_count'].append(q['review_count'])
        results_usc['Location'].append(','.join(q['location']['display_address']))
        results_usc['Coordinates'].append(q['coordinates'])
        results_usc['URL'].append(q['url'])
        results_usc['business_id'].append(q['id'])
    return pd.DataFrame(results_usc)

restaurant_df=pd.DataFrame()
for p in range(len(response_all)):
    #restaurant_df=restaurant_df.append(to_pd_df(response_all[p]['businesses']))
    res_collect=to_pd_df(response_all[p]['businesses'])
    restaurant_df=pd.concat([restaurant_df, res_collect]) #ignore_index=True
restaurant_df=restaurant_df.sort_values(by=['Review_count','Rating'],ascending=(False,False))
restaurant_df=restaurant_df.reset_index(drop=True) #restaurant_df was defined

def review_collection(url):
    try:
        content=requests.get(url)
    except requests.exceptions.ConnectionError as e:
        content="No response"
    soup=BeautifulSoup(content.content,'html.parser')
    review_all=soup.find_all('div', {'class':'review__09f24__oHr9V border-color--default__09f24__NPAKY'})
    results_review={'Name':[],'Text':[],'Length':[],'Pics_Count':[]}
    for review in review_all:
        restaurant_name=soup.find('h1',{'class': 'css-wbh247'}).text
        review_content=review.find('p',{'class': 'comment__09f24__gu0rG css-qgunke'}).text
        review_length=len(review_content)
        try:
            pics_count=review.find('span',{'class': 'css-1oibaro'}).find('a',{'class': 'css-1m051bw'}).text
            pics_count=float(str(pics_count)[0])
        except:
            pics_count=0
        results_review['Name'].append(restaurant_name)
        results_review['Text'].append(review_content)
        results_review['Length'].append(review_length)
        results_review['Pics_Count'].append(pics_count)
    results_df=pd.DataFrame.from_dict(results_review)
    return results_df
        

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print('Print the Yelp_restaurant_USC dataset')
        print(restaurant_df)
        
        print('Print the Yelp_Review dataset')
        combine_df=pd.DataFrame()
        count=0
        for u in restaurant_df['URL']:
            df=review_collection(u)
            combine_df=pd.concat([combine_df,df])
            count+=1
            if count == 1:
                print(str(count)+' restaurant has been collected')
            else:
                print(str(count)+' restaurants have been collected')
        combine_df.reset_index(drop=True)
        print(combine_df)
        
    elif sys.argv[1]=='--scrape':
        print('Print the first N rows of the Yelp_restaurant_USC dataset')
        print(restaurant_df[0:int(sys.argv[2])])
        
        print('Print the first N rows of the Yelp_review dataset')
        combine_df=pd.DataFrame()
        count=0
        for u in restaurant_df['URL']:
            df=review_collection(u)
            combine_df=pd.concat([combine_df,df])
            count+=1
            if count == 1:
                print(str(count)+' restaurant has been collected')
            else:
                print(str(count)+' restaurants have been collected')
        combine_df.reset_index(drop=True)
        print(combine_df[0:int(sys.argv[2])])

    elif sys.argv[1]=='--static':
        print('Save the Yelp_restaurant_USC dataset to the current path')
        restaurant_df.to_csv('Yelp_restaurant_USC.csv')

        print('Save the Yelp_review dataset to the current path')
        count=0
        combine_df=pd.DataFrame()
        for u in restaurant_df['URL']:
            df=review_collection(u)
            combine_df=pd.concat([combine_df,df])
            count+=1
            if count == 1:
                print(str(count)+' restaurant has been collected')
            else:
                print(str(count)+' restaurants have been collected')
        combine_df.to_csv('Yelp_review.csv',index=False) 

else:
    print(f'... Importing module {__name__} ...')
    
print("Now, let's clean the Restaurant_health_score dataset")
health_df_clean=pd.read_csv('Restaurant_health_score.csv')
health_df_clean=health_df_clean.drop_duplicates()
health_df_clean.rename(columns={'ACTIVITY DATE':'Active_Date', 'FACILITY NAME':'Name', 'PROGRAM STATUS':'Status', 'PE DESCRIPTION':'Description', 'FACILITY ADDRESS':'Address', 'FACILITY CITY':'City', 'FACILITY ZIP':'Zip', 'SCORE':'Score'}, inplace=True)
health_df_clean.head()
health_df_clean=health_df_clean.query('City == "LOS ANGELES"')
health_df_clean=health_df_clean.drop(columns='Description')
health_df_clean['Active_Date']=health_df_clean['Active_Date'].astype('datetime64')
health_df_clean=health_df_clean.sort_values('Active_Date', ascending=False).groupby(['Name','Address']).first().reset_index()
health_df_clean.head()
health_df_clean.to_csv('HealthScore_Clean_Data.csv')
print("The dataset was cleaned and saved to the current path with the name 'HealthScore_Clean_Data.csv'")

print("Now, let's add the health score to every restaurant")
res_df=restaurant_df 
res_df2=res_df.copy()
res_df2['Name_begin']=res_df['Name'].map(lambda x:x. split(' ')[0].lower())
res_df2['Address']=res_df['Location'].map(lambda x:x. split(',')[0].lower())
res_df2['Zip']=res_df2['Location'].map(lambda x:x. split(',')[-1][3:9])
res_df2=res_df2.drop(columns=['URL','Coordinates'])
res_df2['Zip']=res_df2['Zip'].astype(int)
res_df2['Name']=res_df2['Name'].str.lower()

hea_df2=health_df_clean.copy()
hea_df2['Name_begin']=hea_df2['Name'].map(lambda x:x. split(' ')[0].lower())
hea_df2['Address']=hea_df2['Address'].str.lower()
hea_df2['Name']=hea_df2['Name'].str.lower()

result_1=pd.merge(hea_df2, res_df2, how='right', on=['Name','Zip'])
result_find=result_1[result_1['Score'].notna()]
result_find=result_find.drop(columns=['Name_begin_x','Address_y','Name_begin_y'])
result_find=result_find.query('Status == "ACTIVE"')
result_2=result_1.drop(result_find.index)
result_2=result_2.drop(columns=['Name_begin_x','Status','City','Score','Address_x'])
result_2.rename(columns={'Name_begin_y':'Name_begin','Address_y':'Address'}, inplace=True)
result_2=pd.merge(hea_df2, result_2, how='right', on=['Name_begin', 'Zip', 'Address'])
result_find2=result_2[result_2['Score'].notna()]
result_find2=result_find2.drop_duplicates()

result_3=result_2.drop(result_find2.index)
result_3=result_3.drop(columns=['Name_x', 'Status','City','Score'])
result_3.rename(columns={'Name_y':'Name'}, inplace=True)
result_3=pd.merge(hea_df2, result_3, how='right', on=['Name_begin', 'Address'])
result_find3=result_3[result_3['Score'].notna()]

result_find_1=result_find[['Name', 'Score','Location','Active_Date']]
result_find2_1=result_find2.loc[:, ('Name_y', 'Score','Location', 'Active_Date_x')]
result_find2_1.rename(columns={'Name_y':'Name','Active_Date_x':'Active_Date'}, inplace=True)
result_find3_1=result_find3.loc[:,('Name_y', 'Score','Location', 'Active_Date')]
result_find3_1.rename(columns={'Name_y':'Name'}, inplace=True)
result_all=pd.concat([result_find_1, result_find2_1,result_find3_1], ignore_index=True)

score_add=pd.merge(result_all, res_df2, how='right', on=['Name','Location'])
score_add=score_add.drop_duplicates()
score_add=score_add.sort_values('Active_Date', ascending=False).groupby(['Name','Address']).head(1).reset_index(drop=True)
score_add=score_add.drop(columns=['Active_Date','Zip','Name_begin','Address'])
score_add.to_csv('Score_add.csv')
print("The dataset combining restaurant information and health score was saved to the current path with the name 'Score_add.csv'")

print("Now, let's begin the data analysis")
print('Following is the text analysis for the reviews')
review_df=combine_df
review_df['Text'] = [i.replace("&amp;amp;", '').replace("\'",'') for i in review_df['Text']]
lan=[detect_langs(i) for i in review_df.Text]
lan=[str(i[0]).split(':')[0] for i in lan]
review_df['lan']=lan
res_rev_1=pd.merge(review_df, restaurant_df, how='left', on=['Name'])

print('Text analysis for high rating restaurants')
high_rate=res_rev_1.query('Rating > 4.5')
stopwords_high=set(stopwords.words('english')+list(ENGLISH_STOP_WORDS)+['com','felt','sure','10','feel','los', 'angeles','recommend','restaurant','came','time','try','order','food','place','come','got','definitely','really','ordered','eat'])
text_all_high=' '.join(high_rate['Text'])
cloud_high=WordCloud(background_color='white', stopwords=stopwords_high).generate(str(text_all_high))
plt.imshow(cloud_high, interpolation='bilinear')
plt.axis('off')
plt.title('The word cloud for high rating restaurants')
plt.show()
vector_high=CountVectorizer(stop_words=stopwords_high, ngram_range=(2,2))
bigrams_high=vector_high.fit_transform(high_rate['Text'])
df_bigram_high=pd.DataFrame(bigrams_high.toarray(), columns=vector_high.get_feature_names_out())
bigram_frequency_high=pd.DataFrame(df_bigram_high.sum(axis=0)).reset_index()
bigram_frequency_high.columns=['bigram', 'frequency']
bigram_frequency_high=bigram_frequency_high.sort_values(by='frequency', ascending=False).head(20)
bigram_frequency_high
fig1=plt.figure(figsize=(15, 5))
plt.barh(bigram_frequency_high['bigram'], bigram_frequency_high['frequency'], color='gray')
plt.xlabel("frequency")
plt.ylabel("Bigram")
plt.title("Bigram with high frquency for high rating restaurants")
plt.show()

print('Text analysis for low rating restaurants.')
low_rate=res_rev_1.query('Rating < 3.5')
stopwords_low=set(stopwords.words('english')+list(ENGLISH_STOP_WORDS)+['com','felt','sure','10','feel','los', 'angeles','recommend','restaurant','came','time','try','order','food','place','come','got','definitely','really','ordered','eat'])
text_all_low=' '.join(low_rate['Text'])
cloud_low=WordCloud(background_color='white', stopwords=stopwords_low).generate(str(text_all_low))
plt.imshow(cloud_low, interpolation='bilinear')
plt.axis('off')
plt.title('The word cloud for low rating restaurants')
plt.show()
vector_low=CountVectorizer(stop_words=stopwords_low, ngram_range=(2,2))
bigrams_low=vector_low.fit_transform(low_rate['Text'])
df_bigram_low=pd.DataFrame(bigrams_low.toarray(), columns=vector_low.get_feature_names_out())
bigram_frequency_low=pd.DataFrame(df_bigram_low.sum(axis=0)).reset_index()
bigram_frequency_low.columns=['bigram', 'frequency']
bigram_frequency_low=bigram_frequency_low.sort_values(by='frequency', ascending=False).head(20)
bigram_frequency_low
fig2=plt.figure(figsize=(15, 5))
plt.barh(bigram_frequency_low['bigram'], bigram_frequency_low['frequency'], color='gray')
plt.xlabel("frequency")
plt.ylabel("Bigram")
plt.title("Bigram with high frquency for low rating restaurants")
plt.show()

print('Following is the correlation analysis between review counts, ratings, and health scores.')
score_df=score_add
score_df=score_df[score_df['Score'].notna()]
score_df=score_df[['Score','Rating','Review_count']]
score_df.corr()

print('The corelation analysis result between rating and health score')
r,p=stats.pearsonr(score_df.Score,score_df.Review_count)
print(f"r-value is {round(r,3)}")
print(f"p-value is {round(p,3)}")
x=score_df['Rating']
y=score_df['Score']
sns.scatterplot(x=x,y=y)
plt.title('The corrleation between rating and health score')    
plt.figure(figsize=(10,5))    
plt.show()

print('The corelation analysis result between review count and health score')
r,p=stats.pearsonr(score_df.Score,score_df.Rating)
print(f"r-value is {round(r,3)}")
print(f"p-value is {round(p,3)}")
x=score_df['Review_count']
y=score_df['Score']
sns.scatterplot(x=x,y=y)
plt.title('The corrleation between review count and health score')    
plt.figure(figsize=(10,5))    
plt.show()

print('Following is the one-way ANOVA analysis between price and review counts') 
price_anova=restaurant_df[['Price', 'Review_count']]
price_anova=price_anova.dropna()
price_anova.groupby(['Price']).mean()
f, p = stats.f_oneway(price_anova[price_anova['Price'] == 1.0].Review_count,
                      price_anova[price_anova['Price'] == 2.0].Review_count,
                      price_anova[price_anova['Price'] == 3.0].Review_count,
                      price_anova[price_anova['Price'] == 4.0].Review_count)
print(f"f-value is {round(f,3)}")
print(f"p-value is {round(p,3)}")
mc = MultiComparison(price_anova['Review_count'], price_anova['Price'])
result = mc.tukeyhsd()
print(result)

print('Following is the OLS regression analysis. The independent variables are review length and picuture counts, and the dependent variable is the rating.')
review_ols1=combine_df[['Name','Length']]
review_ols1_mean=review_ols1.groupby(['Name']).mean()
review_ols2=combine_df[['Name','Pics_Count']]
review_ols2_mean=review_ols2.groupby(['Name']).mean()
review_ols_mean=pd.merge(review_ols1,review_ols2, how='inner', on=['Name'])
res_rev=pd.merge(review_ols_mean, restaurant_df, how='right', on=['Name'])
res_rev=res_rev.drop(columns=['Price'])
res_rev=res_rev.dropna()
res_rev['intercept']=1
lm=sm.OLS(res_rev['Rating'],res_rev[['Length','Pics_Count']])
reg_results=lm.fit()
reg_results.summary()
print(reg_results.summary())