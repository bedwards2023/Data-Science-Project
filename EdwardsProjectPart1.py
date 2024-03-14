"""

Name:Brittany Edwards
Date:
Assignment:Module 9: Project - Part 1
Due Date:
About this project:
All work below was performed by Brittany Edwards

"""

#My two questions I'm attempting to answer
print("What lifestyle factors are significantly associated with the presence of lung cancer?")
print("Do certain lifestyle factors change the stage of the cancer?")

'''
I'm using three data sets:
1.Cancer Prediction Dataset https://www.kaggle.com/datasets/fdcellat/cancer-prediction-dataset
2.Lung Cancer Prediction https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link
3.Smoking related lung cancers https://www.kaggle.com/datasets/raddar/smoking-related-lung-cancers
'''

import pandas as pd
import xlrd
import openpyxl

file2 = r"cancer patient data sets.csv"
file3=r"lung_cancer.csv"

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
print("My y will be in the dataset cancerPredictionDS. It will be either a Yes or No if the person has lung cancer. ")
print("I will using the following attributes from the cancerPatientsDS data frame to answer my questions: ")
print(cancerPatientsDS.columns[4:25].tolist())
print("As well as these attributes from the cancerPredictionDS:")
print(cancerPredictionDS.columns[2:10].tolist())
print("Then for my second question my Y will be the stage of the cancer, in lungcancerDS, also using the same attributes as above. ")