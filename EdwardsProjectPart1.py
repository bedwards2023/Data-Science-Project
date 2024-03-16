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



