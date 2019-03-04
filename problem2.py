problem2 = """Above or Under the Influence?—Like nicotine, the abuse of most substances is correlated with numerous internal
and external factors that affect the likelihood of an individual becoming addicted. Create a model that simulates the
likelihood that a given individual will use a given substance. Take into account social influence and characteristic traits
(e.g., social circles, genetics, health issues, income level, and/or any other relevant factors) as well as characteristics
of the drug itself. Demonstrate how your model works by predicting how many students among a class of 300 high

school seniors with varying characteristics will use the following substances: 
nicotine, marijuana, alcohol, and un-
prescribed opioids."""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

row_format_1 = {'Total':0, 'Male':1, 'Female':2, 'Not Hispanic or Latino':3, 'White':4, 'Black or African American':5, 'American Indian or Alaska Native':6, 'Native Hawaiian or Other Pacific Islander':7, 'Asian':8, 'Two or More Races':9, 'Hispanic or Latino':10, 
                "< High School":11, 'High School Graduate':12, 'Some College/Associate\'s Degree':13, 'College Graduate':14, 'Full-Time':15, 'Part-Time':16, 'Unemployed':17, 'Other':18}
row_format_2 = {'Total':0, 'Northeast':1, 'Midwest':2, 'South':3, 'West':4, 'Large Metro':5, 'Small Metro':6, 'Nonmetro':7, 'Urbanized':8, 'Less Urbanized':9, 'Completely Rural':10,
                'Less Than 100%':11, '100-199%':12, '200% or More':13, 'Private':14, 'Medicaid/CHIP':15, 'Other':16, 'No Coverage':17}

opioid_misues = pd.read_csv('data/Opiod Misues 2017.csv')
marijuana_use = pd.read_csv('data/Marijuana Use 2017.csv')

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
def reformat(mylist):
    reformatted = []
    for x in mylist:
        reformatted.append([x])
    return reformatted

#print(opioid_misues["Aged 12-17 (2017)"].map(lambda x: x.lstrip('*').rstrip('abcd') + '0').astype(float))

attributes_12_17 = {'Nicotine':{
    'Total':0, 'Male':0, 'Female':0, 'Not Hispanic or Latino':0, 'White':0, 'Black or African American':0, 'American Indian or Alaska Native':0, 'Native Hawaiian or Other Pacific Islander':0, 'Asian':0, 'Two or More Races':0, 'Hispanic or Latino':0, 
    "< High School":0, 'High School Graduate':0, 'Some College/Associate\'s Degree':0, 'College Graduate':0, 'Full-Time':0, 'Part-Time':0, 'Unemployed':0, 'Other':0, 'Northeast':1, 'Midwest':2, 'South':3, 'West':4, 'Large Metro':5, 'Small Metro':6, 
    'Nonmetro':7, 'Urbanized':8, 'Less Urbanized':9, 'Completely Rural':10, 'Less Than 100%':11, '100-199%':12, '200% or More':13, 'Private':14, 'Medicaid/CHIP':15, 'Other':16, 'No Coverage':17
}, 'Marijuana':{
    'Total':0, 'Male':0, 'Female':0, 'Not Hispanic or Latino':0, 'White':0, 'Black or African American':0, 'American Indian or Alaska Native':0, 'Native Hawaiian or Other Pacific Islander':0, 'Asian':0, 'Two or More Races':0, 'Hispanic or Latino':0, 
    "< High School":0, 'High School Graduate':0, 'Some College/Associate\'s Degree':0, 'College Graduate':0, 'Full-Time':0, 'Part-Time':0, 'Unemployed':0, 'Other':0, 'Northeast':1, 'Midwest':2, 'South':3, 'West':4, 'Large Metro':5, 'Small Metro':6, 
    'Nonmetro':7, 'Urbanized':8, 'Less Urbanized':9, 'Completely Rural':10, 'Less Than 100%':11, '100-199%':12, '200% or More':13, 'Private':14, 'Medicaid/CHIP':15, 'Other':16, 'No Coverage':17
}, 'Alchohol':{
    'Total':0, 'Male':0, 'Female':0, 'Not Hispanic or Latino':0, 'White':0, 'Black or African American':0, 'American Indian or Alaska Native':0, 'Native Hawaiian or Other Pacific Islander':0, 'Asian':0, 'Two or More Races':0, 'Hispanic or Latino':0, 
    "< High School":0, 'High School Graduate':0, 'Some College/Associate\'s Degree':0, 'College Graduate':0, 'Full-Time':0, 'Part-Time':0, 'Unemployed':0, 'Other':0, 'Northeast':1, 'Midwest':2, 'South':3, 'West':4, 'Large Metro':5, 'Small Metro':6, 
    'Nonmetro':7, 'Urbanized':8, 'Less Urbanized':9, 'Completely Rural':10, 'Less Than 100%':11, '100-199%':12, '200% or More':13, 'Private':14, 'Medicaid/CHIP':15, 'Other':16, 'No Coverage':17
}, 'Opioid':{
    'Total':0, 'Male':0, 'Female':0, 'Not Hispanic or Latino':0, 'White':0, 'Black or African American':0, 'American Indian or Alaska Native':0, 'Native Hawaiian or Other Pacific Islander':0, 'Asian':0, 'Two or More Races':0, 'Hispanic or Latino':0, 
    "< High School":0, 'High School Graduate':0, 'Some College/Associate\'s Degree':0, 'College Graduate':0, 'Full-Time':0, 'Part-Time':0, 'Unemployed':0, 'Other':0, 'Northeast':1, 'Midwest':2, 'South':3, 'West':4, 'Large Metro':5, 'Small Metro':6, 
    'Nonmetro':7, 'Urbanized':8, 'Less Urbanized':9, 'Completely Rural':10, 'Less Than 100%':11, '100-199%':12, '200% or More':13, 'Private':14, 'Medicaid/CHIP':15, 'Other':16, 'No Coverage':17
}}


def reformat_data(column, dict, keys_to_index = row_format_1):
    for i, nr in enumerate(column):
        tp = list(keys_to_index.keys())[list(keys_to_index.values()).index(i)]
        dict[tp] = nr

def clean_column(column):
    try:
        return column.map(lambda x: '0' + x.lstrip('*abcd').rstrip('*abcd')).astype(float)
    except:
        return column
age = 'Aged 12-17 (2017)'
nicotine = clean_column(pd.read_csv('data/Nicotine Use 2017.csv')[age])
reformat_data(nicotine, attributes_12_17['Nicotine'])

nicotine = clean_column(pd.read_csv('data/Marijuana Use 2017.csv')[age])
reformat_data(nicotine, attributes_12_17['Marijuana'])

nicotine = clean_column(pd.read_csv('data/Alchohol Use 2017.csv')[age])
reformat_data(nicotine, attributes_12_17['Alchohol'])

nicotine = clean_column(pd.read_csv('data/Opiod Misues 2017.csv')[age])
reformat_data(nicotine, attributes_12_17['Opioid'])

nicotine = clean_column(pd.read_csv('data/Nicotine Use 2017 SES.csv')[age])
reformat_data(nicotine, attributes_12_17['Nicotine'], row_format_2)

nicotine = clean_column(pd.read_csv('data/Marijuana Use 2017 SES.csv')[age])
reformat_data(nicotine, attributes_12_17['Marijuana'], row_format_2)

nicotine = clean_column(pd.read_csv('data/Alchohol Use 2017 SES.csv')[age])
reformat_data(nicotine, attributes_12_17['Alchohol'], row_format_2)

nicotine = clean_column(pd.read_csv('data/Opioid Misues 2017 SES.csv')[age])
reformat_data(nicotine, attributes_12_17['Opioid'], row_format_2)

print(attributes_12_17)

def reformatRows(df):
    """assumes all variables are within a single data frame and each variable has its own column"""
    mydata = []
    for row in df.iterrows():
        current = []
        for name, value in row.iteritems():
            current.append(value)
        mydata.append(current)
    return mydata

from sklearn import tree
X = reformatRows()
Y = 
line = linear_model.Linear_Regression()
line.fit(X,Y)
logistic = linear_model.LogisticRegression()
me = tree.DecisionTreeClassifier()
me.fit(X, Y)

