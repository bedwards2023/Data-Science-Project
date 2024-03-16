"""

Name:Brittany Edwards
Date:
Assignment:Module 9: Project - Part 1
Due Date:
About this project:
All work below was performed by Brittany Edwards

"""

#My two questions I'm attempting to answer
print("***********************THE QUESTIONS*************************************************")
print("What lifestyle factors are significantly associated with the presence of lung cancer?")
print("Are there symptoms that can predict if a person has lung cancer? ")
print("*************************************************************************************")
'''
I'm using three data sets:
1.Cancer Prediction Dataset https://www.kaggle.com/datasets/fdcellat/cancer-prediction-dataset
2.Lung Cancer Prediction https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link
3.Lung Cancer https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer
'''

import pandas as pd
import xlrd
import openpyxl
import scipy
from scipy import stats
import statistics
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import string
import contractions
import re
from collections import Counter


'''*********FUNCTIONS HERE****************'''
#I used the function from the example video to print the desired values for each numeric column
def NumericAttributePrint(dataframe,attribute):
    print("**********************")
    print(attribute)
    print('Possible values - ', dataframe[attribute].unique())
    print('value counts - ')
    print(dataframe[attribute].value_counts())
    print('Max - ', dataframe[attribute].max())
    print('Min - ', dataframe[attribute].min())
    print('Central Tendency')
    print('Mean - ', dataframe[attribute].mean())
    print('Median - ', dataframe[attribute].median())
    print('Geometric mean - ', scipy.stats.gmean(dataframe[attribute]))
    print('Variance Metric')
    print('Standard Deviation - ', statistics.stdev(dataframe[attribute]))

#This function will be called to print the possible values and value counts for any
#nonnumeric column
def NonNumericPrint(dataframe,attribute):
    print("**********************")
    print(attribute)
    print('Possible values - ', dataframe[attribute].unique())
    print('value counts - ')
    print(dataframe[attribute].value_counts())


def preprocess_data(dataframe):
    rows_before = dataframe.shape[0]
    dataframe.dropna(inplace=True)

    rows_after = dataframe.shape[0]

    if rows_before != rows_after:
        print(f"{rows_before - rows_after} rows were dropped due to missing values.")
    return dataframe

#This will calculate their lifestly health score
def calculate_lifestyle_score(row):
    score = 0
    for attr, weight in attribute_weights.items():
        if pd.notna(row[attr]) and pd.to_numeric(row[attr], errors='coerce') >= 1:
            score += pd.to_numeric(row[attr], errors='coerce') * weight
    return score

#This will assign their lifestyle score to a rank
def assign_lifestyle_rank(score):
    if score <= 8:
        return 'Very Healthy'
    elif score > 8 and score <= 16 :
        return 'Healthy'
    elif score > 16 and score <= 24:
        return 'Moderate'
    elif score > 24:
        return 'Unhealthy'

def tokenization(text):
    tokens = text.split()
    return tokens
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree


file2 = r"cancer patient data sets.csv"
file3=r"survey lung cancer.csv"


'''
For each file
Using python, create a script that imports the data set and evaluate the central tendencies and variation of attributes. '''
cancerPredictionDS = pd.read_excel(r'Cancer_prediction_dataset.xlsx', engine='openpyxl')
cancerPatientsDS = pd.read_csv(file2)
lungcancerDS = pd.read_csv(file3)

print(cancerPredictionDS.describe())
print(cancerPatientsDS.describe())
print(lungcancerDS.describe())

'''
List the data sets and attributes that could be used to answer each of the 2 questions you identified, namely, which attribute are you going to select to be the "y" attribute (value your model will predict), 
and which attributes are you going to select to be your "X" attributes (the attributes you will base your prediction upon). 
Note: You must have at least 15 X attributes and some of these attributes must be strings (i.e., text input)
'''
print("*************************************************************************************")
print("For my first question, my y will be in the dataset cancerPredictionDS. It will be either a Yes or No if the person has lung cancer. ")
print("I will using the lifestyle attributes from cancerPredictionDS: Marital Status, Children, Smoker, Employed, Years Worked,Income Level,Social Media,Online Gaming ")
print("And from cancerPatientsDS: air pollution exposure, alcohol use, dust allergy, occupational hazards, genetic risk, chronic lung disease, balanced diet, obesity, smoking, passive smoker")
print("For my second question my Y is still in the dataset cancerPredictionDS.")
print("But my x's will be symptom attributes from cancerPatientsDS: chest pain, coughing of blood, fatigue, weight loss ,shortness of breath ,wheezing ,swallowing difficulty ,clubbing of finger nails and snoring")
print("And from lungcancerDS: Yellow fingers,Anxiety,Chronic Disease,Fatigue,Wheezing,Coughing,Shortness of Breath,Swallowing Difficulty,Chest pain")


'''
Combine the 3 dataframes into a single dataframe by joining on matching attributes  '''

#First I need to make some changes to my cancerPatientsDS because Gender is a 1 or 2 value, not a string 'Male' or 'Female' and I want to join my first two datasets on Age and Gender
gender_mapping = {1: 'Male', 2: 'Female'}
cancerPatientsDS['Gender'] = cancerPatientsDS['Gender'].replace(gender_mapping)
#I had to do the same in the lungcancerDS because Gender was either 'M' or 'F'
gender_mapping2 = {'M': 'Male', 'F': 'Female'}
lungcancerDS['GENDER'] = lungcancerDS['GENDER'].replace(gender_mapping2)
#And then I had to change the column names in lungcancerDS because the names were all capitilized.
column_name_mapping = {'GENDER': 'Gender', 'AGE': 'Age'}
lungcancerDS = lungcancerDS.rename(columns=column_name_mapping)


#Im merging all my datasets on Age and Gender because it's in all three datasets.
merged_df = pd.merge(cancerPredictionDS, cancerPatientsDS, on=['Age', 'Gender'], how='inner')
finaldf = pd.merge(merged_df,lungcancerDS, on=['Age','Gender'], how ='inner')
print(finaldf)



'''
CENTRAL TENDENCIES 
Create data frames for each of the questions that only include the attributes that you identified above to match the attribute(s) you are looking to predict (y value) and the attributes that you will base your prediction upon 
For each numeric attribute displays the: 
Possible values
value counts
Name and dtype
Max
Min
Mean
Median
Geometric mean
Standard Deviation
For each non-numeric attribute displays the:
Possible values
value counts
'''
#These are my two new dataframes based on the attributes that will help answer my question. Its lifestyle vs symptoms
lifestyledf = finaldf[['Age','Gender','Marital Status','Children','Smoker','Employed','Years Worked','Income Level','Social Media','Online Gaming','Air Pollution','Alcohol use','OccuPational Hazards','Balanced Diet','Obesity','Passive Smoker','ANXIETY','Cancer']]
symptomdf = finaldf[['Age','Gender','chronic Lung Disease','Chest Pain','Coughing of Blood','Fatigue','Weight Loss','Shortness of Breath','Wheezing','Swallowing Difficulty','Clubbing of Finger Nails','ANXIETY','CHRONIC DISEASE','ALLERGY ','COUGHING','CHEST PAIN','Cancer']]


for column in lifestyledf.columns:
    if pd.api.types.is_numeric_dtype(lifestyledf[column]):
        print("Numeric Value")
        NumericAttributePrint(lifestyledf,column)
    else:
        print("NonNumeric Value")
        NonNumericPrint(lifestyledf,column)

for column in symptomdf.columns:
    if pd.api.types.is_numeric_dtype(symptomdf[column]):
        print("Numeric Value")
        NumericAttributePrint(symptomdf,column)
    else:
        print("NonNumeric Value")
        NonNumericPrint(symptomdf,column)

'''Data Wrangling & Scores and Rankings (30 points)

Using the data frames created for each question above add code that replaces na or invalid values in for each attribute to the mean of the attribute or remove such row '''

for column in lifestyledf.columns:
    preprocess_data(lifestyledf)

for column in symptomdf.columns:
    preprocess_data(symptomdf)



'''
Use a function to create a range based upon an attribute in one of these dataframes
'''

#I'm going to rank their lifestyle choices to see if they're healthy or not

#I gave each attribute a weight
attribute_weights = {
    'Smoker': 1,
    'Obesity': 1,
    'Passive Smoker': 1,
    'Alcohol use': 1,
    'OccuPational Hazards': 1,
    'Air Pollution': 1,
    'Online Gaming': 1,
    'Social Media': 1,
    'Balanced Diet': -1
}

'''Add an attribute to your dataframe using this function that you just created'''
#Then I calculate the score of their lifestyle
lifestyledf['Lifestyle Score'] = lifestyledf.apply(calculate_lifestyle_score, axis=1)
#Then assign that score to a ranking
lifestyledf['Lifestyle Rank'] = lifestyledf['Lifestyle Score'].apply(assign_lifestyle_rank)

print(lifestyledf)

'''
Create a scatter plot based upon this new attribute and another attribute in this dataframe'''
# Create two separate DataFrames for individuals with and without cancer
df_with_cancer = lifestyledf[lifestyledf['Cancer'] == 'Yes']
df_without_cancer = lifestyledf[lifestyledf['Cancer'] == 'No']

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df_with_cancer['Age'], df_with_cancer['Lifestyle Score'], color='red', label='With Cancer', alpha=0.5)
plt.scatter(df_without_cancer['Age'], df_without_cancer['Lifestyle Score'], color='blue', label='Without Cancer', alpha=0.5)
plt.title('Scatter Plot of Lifestyle Score vs Age with Cancer')
plt.xlabel('Age')
plt.ylabel('Lifestyle Score')
plt.legend()
plt.show()


'''Text Attributes (30 points)

Compute and display the 20 most common text in one of the text based attributes
Parse the data frame based upon the text based attribute containing some common text 
Write python script that counts the number of entries that contain some common text by another attribute in the dataframe of your choosing
Use matplotlib to generate python script that displays a graph of this data'''

#For this part I consulted with Dr.Works and she said I could use another data set because my datasets used above dont work with this.

#I'll be using a movie review data set from kaggle https://www.kaggle.com/datasets/nltkdata/movie-review?select=movie_review.csv


file4 = r'movie_review.csv'
moviereviewdf = pd.read_csv(file4)

#clean the data first
moviereviewdf['clean_text']= moviereviewdf['text'].apply(lambda x: x.lower())
moviereviewdf['clean_text']= moviereviewdf['clean_text'].apply(lambda x:remove_punctuation(x))
moviereviewdf['clean_text']= moviereviewdf['clean_text'].apply(lambda x: re.sub('\w*\d\w*','', x))
moviereviewdf['clean_text']= moviereviewdf['clean_text'].apply(lambda x: contractions.fix(x))

textdf = moviereviewdf['clean_text']
p = Counter(" ".join(textdf).split()).most_common(20)
rslt = pd.DataFrame(p, columns=['Word', 'Frequency'])

#Compute and display the 20 most common text in one of the text based attributes
print(rslt)

common_word = rslt.iloc[0]['Word']

filtered_df = moviereviewdf[moviereviewdf['clean_text'].str.contains(common_word)]
#Write python script that counts the number of entries that contain some common text by another attribute in the dataframe of your choosing
word_counts_by_sentiment = filtered_df.groupby('tag').size()

#Use matplotlib to generate python script that displays a graph of this data
#So I'm showing the the sentiment of the where the common words appeared in the reviews
plt.bar(word_counts_by_sentiment.index, word_counts_by_sentiment.values)
plt.xlabel('Sentiment')
plt.ylabel('Frequency of Common Word')
plt.title(f'Frequency of "{common_word}" by Sentiment')
plt.show()

'''Variance, Covariance, and Correlation (40 points)

Compute and display the variance of each "X" attributes  
Compute and display the covariance of each "X" attribute and "y" attribute in each dataset
You must write two sentences for each covariance that describe the type of the covariance (positive, negative, or none) and what that means in terms of the two attributes. 
Compute and display the correlation of each "X" attribute and "y" attribute in each dataset
You must write two sentences for each correlation that describe the type of the covariance (positive, negative, or none), the degree (See https://statisticsbyjim.com/basics/correlations/) and what that means in terms of the two attributes.'''

#Compute and display the variance of each "X" attributes
print(lifestyledf.var(numeric_only=True))
print(symptomdf.var(numeric_only=True))

#Compute and display the covariance of each "X" attribute and "y" attribute in each dataset

#So first I have to do some housekeeping with my first lifestyle data frame. There are some string columns I need to change to numeric values,
#So I'm going to make a copy of the dataframe and change the strings to numeric values.
cancermapping = {'Yes': 1, 'No': 2}
covariance_lifestyledf = lifestyledf
covariance_lifestyledf['Cancer'] = covariance_lifestyledf['Cancer'].replace(cancermapping)
covariance_lifestyledf['Employed'] = covariance_lifestyledf['Employed'].replace(cancermapping)
covariance_lifestyledf['Smoker'] = covariance_lifestyledf['Smoker'].replace(cancermapping)
covariance_lifestyledf['Online Gaming'] = covariance_lifestyledf['Online Gaming'].replace(cancermapping)
covariance_lifestyledf['Social Media'] = covariance_lifestyledf['Social Media'].replace(cancermapping)
cov_matrix = covariance_lifestyledf.cov(numeric_only=True)
expanded_output = cov_matrix.to_string()
print("Covariance of the lifestyle dataframe ")
print("****************Lifestyle covariance******************")
print(expanded_output)
'''For context the cancer patient data sets which mainly make up my x's are various lifestyle factors that have a rating 1-8 or 1-7."
"I interperted the data as, 1 is the highest rating. So if Air Pollution is rated 1 then its the worst and the higher ratings, 8 or 7 would"
"be the least Air Pollution."
"Also, some of my attributes are binary as well. When the covariance of two binary attributes are positive that means x=1 and y=1."
"From my research when you're looking at the covariance of a continuous value and a binary value, when the covariance is positive that means as the continious"
"value increase y=1 and when the covariance is negative, as the continious value grows y=0."
"Age:  -0.755709 Negative. So this is saying as the Age variable increases the rate of y=1 decreases. There might be an age that a person reaches where their"
"chances of developing lung cancer decreases."
"Children: 0.005849 Positive. This is a really small number. The relationship between children and cancer rates are really weak so having kids wont give you lung cancer."
"Smoker:  0.018622 Positive. Smoker was a binary attribute in my data set. So when Smoker =1 so does Cancer. This makes sense because we're looking at lung cancer."
"Employed:  0.047881 Positive. This is a binary value as well and positive, so when employed =1 so does cancer. This is a really small number though so I wouldn't use this lifestyle attribute."
"Years Worked: -1.446028 Negative. This was the highest covariance out of all of them, surprisingly. So when the number of years worked increases the rate of cancer =1 decreases. I guess this ties into age."
"Social Media: 0.046588 Positive. This was one of the rating attributes, so when social media increases that's actually saying their social media usage is lower. So a rating around 7-8 would mean cancer =1."
"It could be implied that when social media usage is lower, the rate of cancer=1, however the covariance is too small to be reliable."
"Online Gaming: -0.078541 Negative. As online rates increase, so they get to ranking 7-8, then cancer=0. So this is implying that lower online gaming means lower lung cancer rates."
"Air Pollution: -0.104843 Negative. When someone ranked their air pollution as 1 then cancer rates increases.  This makes sense based on common knowledge of lung cancer."
"Alcohol use:  -0.199225 Negative. So if someone ranked their alcohol use as 1-3 then the rate of cancer =1 increases. This is the second highest covariance, so there is something to be said about"
"alcohol use and lung cancer."
"OccuPational Hazards: -0.170237 Negative. People rating their occupational hazards as 1-3 appear to have more rates of lung cancer. They could be more exposed to smoke and pollutants so that makes sense."
"Balanced Diet: -0.089079 Negative. This one doesnt make a ton of sense. A person ranking their diet as 1 would have more cancer=1, however the covariance is small so I guess it doesnt matter."
"Obesity: -0.095557 Negative. Someone rating their obesity as 1-3 would have more cancer =1. Not a very strong relationship."
"Passive Smoker:  -0.094979 Negative. Not sure what a passive smoker is from the description, but when someone is ranked as 1-3 they have more cancer =1"
"ANXIETY: -0.026633 Negative. Not a very strong relationship."
"Lifestyle Score:-0.575762 Negative This was my scoring of their overall lifestyle health. The relationship is strong but this is implying that someone"
"with a poor lifestyle score would have less chances of lung cancer.'''

covariance_symptomdf = symptomdf
covariance_symptomdf['Cancer'] = covariance_symptomdf['Cancer'].replace(cancermapping)
cov_matrix = covariance_symptomdf.cov(numeric_only=True)
expanded_output = cov_matrix.to_string()
print("Covariance of the Symptom Dataframe ")
print(expanded_output)

'''Again for this the attributes are mainly rankings, where 1= the worst/most and the lower rankings means less.
Age:  -0.755709 Negative. As age increase the rates of cancer =1 decreases. 
chronic Lung Disease: -0.097766 Negative. Someone rating their chronic lung disease as 1 would have more cancer =1.
Chest Pain:  -0.113134 Negative. This was the biggest covariance from the list, implying a strong relationship. Someone rating their 
chest pain as 1 would have more cancer=1
Coughing of Blood:   -0.073318 Negative. Not a terribly strong relationship but would imply rating =1 equals more cancer =1
Fatigue:  -0.054619 Negative. Not a very strong relationship between a rating 1 Fatigue and lung cancer.
Weight Loss:  -0.029699 Negative. Stronger relationship than others. 
Shortness of Breath: -0.085684 Negative. Someone rating their shortness of breath to be a 1
would have more lung cancer rates. 
Wheezing:  -0.039455 Negative. Stronger relationship than other symptoms. This could be a useful symptom to use.
Swallowing Difficulty: 0.022651 Positive. Someone rating their swallowing difficulty would actually have cancer=0 due to it being positive. 
Clubbing of Finger Nails: -0.092374 Negative. Clubbing of finernails is actually used as a indicator of lung disease but it doesnt seem as strong as other symptoms 
ANXIETY:  -0.026633 Negative. People rating their anxiety high could have more rates of lung cancer. They could be smoking due to their anxiety.
CHRONIC DISEASE:  0.007942 Positive. Very small number, not a strong relationship. 
ALLERGY:   0.010062 Positive. Also a very small number, not a strong relationship.
COUGHING:  0.013037 Positive. Not a strong relationship.
CHEST PAIN:    0.002127 Positive. Even weaker relationship. 
'''
