problem2 = """Above or Under the Influence?—Like nicotine, the abuse of most substances is correlated with numerous internal
and external factors that affect the likelihood of an individual becoming addicted. Create a model that simulates the
likelihood that a given individual will use a given substance. Take into account social influence and characteristic traits
(e.g., social circles, genetics, health issues, income level, and/or any other relevant factors) as well as characteristics
of the drug itself. Demonstrate how your model works by predicting how many students among a class of 300 high school seniors with varying characteristics will use the following substances: 
nicotine, marijuana, alcohol, and un-
prescribed opioids."""

# we assume our individuals exist in 2017

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random

row_format_1 = {'Total':0, 'Male':1, 'Female':2, 'Not Hispanic or Latino':3, 'White':4, 'Black or African American':5, 'American Indian or Alaska Native':6, 'Native Hawaiian or Other Pacific Islander':7, 'Asian':8, 'Two or More Races':9, 'Hispanic or Latino':10, 
                "< High School":11, 'High School Graduate':12, 'Some College/Associate\'s Degree':13, 'College Graduate':14, 'Full-Time':15, 'Part-Time':16, 'Unemployed':17, 'Other':18}
row_format_2 = {'Total':0, 'Northeast':1, 'Midwest':2, 'South':3, 'West':4, 'Large Metro':5, 'Small Metro':6, 'Nonmetro':7, 'Urbanized':8, 'Less Urbanized':9, 'Completely Rural':10,
                'Less Than 100%':11, '100-199%':12, '200% or More':13, 'Private':14, 'Medicaid/CHIP':15, 'Other':16, 'No Coverage':17}

# opioid_misuse = pd.read_csv('data/Opioid Misuse 2017.csv')
# marijuana_use = pd.read_csv('data/Marijuana Use 2017.csv')

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
def reformat(mylist):
    reformatted = []
    for x in mylist:
        reformatted.append([x])
    return reformatted

#print(opioid_misuse["Aged 12-17 (2017)"].map(lambda x: x.lstrip('*').rstrip('abcd') + '0').astype(float))

attributes_12_17 = {'Nicotine':{
    'Total':0, 'Male':0, 'Female':0, 'Not Hispanic or Latino':0, 'White':0, 'Black or African American':0, 'American Indian or Alaska Native':0, 'Native Hawaiian or Other Pacific Islander':0, 'Asian':0, 'Two or More Races':0, 'Hispanic or Latino':0, 
    "< High School":0, 'High School Graduate':0, 'Some College/Associate\'s Degree':0, 'College Graduate':0, 'Full-Time':0, 'Part-Time':0, 'Unemployed':0, 'Other':0, 'Northeast':1, 'Midwest':2, 'South':3, 'West':4, 'Large Metro':5, 'Small Metro':6, 
    'Nonmetro':7, 'Urbanized':8, 'Less Urbanized':9, 'Completely Rural':10, 'Less Than 100%':11, '100-199%':12, '200% or More':13, 'Private':14, 'Medicaid/CHIP':15, 'Other':16, 'No Coverage':17
}, 'Marijuana':{
    'Total':0, 'Male':0, 'Female':0, 'Not Hispanic or Latino':0, 'White':0, 'Black or African American':0, 'American Indian or Alaska Native':0, 'Native Hawaiian or Other Pacific Islander':0, 'Asian':0, 'Two or More Races':0, 'Hispanic or Latino':0, 
    "< High School":0, 'High School Graduate':0, 'Some College/Associate\'s Degree':0, 'College Graduate':0, 'Full-Time':0, 'Part-Time':0, 'Unemployed':0, 'Other':0, 'Northeast':1, 'Midwest':2, 'South':3, 'West':4, 'Large Metro':5, 'Small Metro':6, 
    'Nonmetro':7, 'Urbanized':8, 'Less Urbanized':9, 'Completely Rural':10, 'Less Than 100%':11, '100-199%':12, '200% or More':13, 'Private':14, 'Medicaid/CHIP':15, 'Other':16, 'No Coverage':17
}, 'Alcohol':{
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

marijuana = clean_column(pd.read_csv('data/Marijuana Use 2017.csv')[age])
reformat_data(marijuana, attributes_12_17['Marijuana'])

alcohol = clean_column(pd.read_csv('data/Alcohol Use 2017.csv')[age])
reformat_data(alcohol, attributes_12_17['Alcohol'])

opioid = clean_column(pd.read_csv('data/Opioid Misuse 2017.csv')[age])
reformat_data(opioid, attributes_12_17['Opioid'])

nicotine_ses = clean_column(pd.read_csv('data/Nicotine Use 2017 SES.csv')[age])
reformat_data(nicotine_ses, attributes_12_17['Nicotine'], row_format_2)

marijuana_ses = clean_column(pd.read_csv('data/Marijuana Use 2017 SES.csv')[age])
reformat_data(marijuana_ses, attributes_12_17['Marijuana'], row_format_2)

alcohol_ses = clean_column(pd.read_csv('data/Alcohol Use 2017 SES.csv')[age])
reformat_data(alcohol_ses, attributes_12_17['Alcohol'], row_format_2)

opioid_ses = clean_column(pd.read_csv('data/Opioid Misuse 2017 SES.csv')[age])
reformat_data(opioid_ses, attributes_12_17['Opioid'], row_format_2)

# print(attributes_12_17)

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
# X = reformatRows()
# Y = 
# line = linear_model.Linear_Regression()
# line.fit(X,Y)
# logistic = linear_model.LogisticRegression()
# me = tree.DecisionTreeClassifier()
# me.fit(X, Y)


# labels = {
#     'male': 'Male', 'female': 'Female', 'not hispanic': 'Not Hispanic or Latino', 'white': 'White',
#     'african american': 'Black or African American', 'native american': 'American Indian or Alaska Native',
#     'pacific islander': 'Native Hawaiian or Other Pacific Islander', 'asian': 'Asian',
#     'mexican': 'Hispanic or Latino','puerto rican': 'Hispanic or Latino', 'cuban': 'Hispanic or Latino', 'other hispanic': 'Hispanic or Latino'
# }
    #'Northeast':1, 'Midwest':2, 'South':3, 'West':4, 'Large Metro':5, 'Small Metro':6, 
    #'Nonmetro':7, 'Urbanized':8, 'Less Urbanized':9, 'Completely Rural':10, 'Less Than 100%':11, '100-199%':12, '200% or More':13, 'Private':14, 'Medicaid/CHIP':15, 'Other':16, 'No Coverage':17
# demographics = {}

# dem_data = pd.read_csv("data/Demographics.csv")
# keys = dem_data[dem_data.columns[0]]
# for i in range(len(dem_data['2017'])):
#     demographics[labels[keys[i]]] = dem_data['2017'][i]
# 15.8% seniors

age_groups = {'12': 'Aged 12-17 (2017)', '13': 'Aged 12-17 (2017)', '14': 'Aged 12-17 (2017)', '15': 'Aged 12-17 (2017)', '16': 'Aged 12-17 (2017)', '17': 'Aged 12-17 (2017)', '18': 'Aged 18-25 (2017)'}
drugs  = attributes_12_17#{'nicotine': nicotine_data}
drugs_by_age_data = pd.read_csv("data/use_by_age_2016_2017.csv")
labels = {'Nicotine': 'tobacco misuse in past year (2017)', 'Alcohol': 'alcohol past year (2017)', 'Marijuana': 'marijuana past year (2017)', 'Opioid': 'opioids past year (2017)'}
drugs_by_age = {}
age_column = drugs_by_age_data[drugs_by_age_data.columns[0]]
for drug in labels:
    drugs_by_age[drug] = {}
    for i in range(len(age_column)):
        drugs_by_age[age_column[i]] = drugs_by_age_data[labels[drug]][i]

def predict(drug, person):
    drug_data = drugs[drug]
    p_total = float(drugs_by_age[person['age']])/100 # begin with a generic person in the age group with no traits
    for trait in person:
        if trait == 'age' or trait == 'age_group': # age is linked to every other trait so we ignore it
            continue
        # p_of_trait = float(person[trait])/100
        age_group = age_groups[person['age']]
        p_age_group_average = float(drug_data['Total'])/100
        p_age_group_trait = float(drug_data[person[trait]])/100
        p_total *= p_age_group_trait / p_age_group_average # modify probability by the prevalence of drug usae by people with the trait
        # print(p_age_group_trait / p_age_group_average)
    return p_total

traits = {
        'sex': ['Male', 'Female'] , 'hispanic': ['Hispanic or Latino', 'Not Hispanic or Latino'],
        'race': ['White', 'Black or African American', 'American Indian or Alaska Native', 'Native Hawaiian or Other Pacific Islander', 'Asian'],
        'region': ['Northeast',  'Midwest', 'South', 'West'],
        'metro': ['Large Metro', 'Small Metro', 'Nonmetro'],
        'urban': ['Urbanized', 'Less Urbanized', 'Completely Rural'],
        'poverty': ['Less Than 100%', '100-199%', '200% or More'],
        'healthcare': ['Private', 'Medicaid/CHIP', 'Other', 'No Coverage']
    }

def generate_person(age):
    person = {'age': str(age)}
    for trait in traits:
        person[trait] = random.choice(traits[trait])
    return person

def seniors():
    age = 17
    predictions = {drug: 0 for drug in drugs}
    for i in range(300):
        person = generate_person(age)
        for drug in drugs:
            p_drug = predict(drug, person)
            predictions[drug] += p_drug
    return predictions

print(seniors())
        