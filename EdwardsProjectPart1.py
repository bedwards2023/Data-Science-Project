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
print("My y will be in the dataset cancerPredictionDS. It will be either a Yes or No if the person has lung cancer. ")
print("I will using the following attributes from the cancerPatientsDS data frame to answer my questions: ")
print(cancerPatientsDS.columns[4:25].tolist())
print("As well as these attributes from the cancerPredictionDS:")
print(cancerPredictionDS.columns[2:10].tolist())
print("Then for my second question my Y will also be if the person has lung cancer or not, ")
print("*************************************************************************************")
'''
Combine the 3 dataframes into a single dataframe by joining on matching attributes  '''

#First I need to make some changes to my cancerPatientsDS because Gender is a 1 or 2 value, not a string 'Male' or 'Female' and I want to join my first two datasets on Age and Gender
gender_mapping = {1: 'Male', 2: 'Female'}

cancerPatientsDS['Gender'] = cancerPatientsDS['Gender'].replace(gender_mapping)
print(cancerPatientsDS)
gender_mapping2 = {'M': 'Male', 'F': 'Female'}
lungcancerDS['GENDER'] = lungcancerDS['GENDER'].replace(gender_mapping2)
column_name_mapping = {'GENDER': 'Gender', 'AGE': 'Age'}
lungcancerDS = lungcancerDS.rename(columns=column_name_mapping)



merged_df = pd.merge(cancerPredictionDS, cancerPatientsDS, on=['Age', 'Gender'], how='inner')
print(merged_df)

finaldf = pd.merge(merged_df,lungcancerDS, on=['Age','Gender'], how ='inner')
print(finaldf)



'''
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