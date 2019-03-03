import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
#import sklearn as skl
high_school_drug_use = pd.read_csv("NIH-DrugTrends-DataSheet.csv", header = 0)
#print(high_school_drug_use['8th Graders 2015'])
#print(tabulate(high_school_drug_use, headers='keys', tablefmt='psql'))

#graphing libraries

import matplotlib
import matplotlib.pyplot as plt

years = ['2015', '2016', '2017', '2018']
grades = ['8th Graders', '10th Graders', '12th Graders']
def get_gradeyear(year, grade):
    return high_school_drug_use[grade + " " + year]

drugs = {}
last_drug = ''
for i, drug in enumerate(high_school_drug_use['Drug']):
    if type(drug) == str:
        last_drug = drug
        drugs[drug] = {}
    drugs[last_drug][high_school_drug_use['Time Period'][i]] = i
# print("drugs", drugs)

drug = drugs['Any Vaping']['Past Month']
print(get_gradeyear('2015', '8th Graders'))

data_points = []
grade = '8th Graders'
for year in years:
    data_point = get_gradeyear(year, grade)[drug]
    if data_point[0] == '[' and data_point[-1] == ']':
        data_points.append(float(data_point[1:-1]))
    elif data_point == '-':
        data_points.append(0)
    else:
        data_points.append(float(data_point))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(years, data_points)
# fig.savefig('test.svg')
plt.show()

#problem statements

problem1 = """Darth Vapor—Often containing high doses of nicotine, vaping (inhalation of an aerosol created by vaporizing
a liquid) is hooking a new generation that might otherwise have chosen not to use tobacco products. Build a
mathematical model that predicts the spread of nicotine use due to vaping over the next 10 years. Analyze how the
growth of this new form of nicotine use compares to that of cigarettes."""

problem2 = """Above or Under the Influence?—Like nicotine, the abuse of most substances is correlated with numerous internal
and external factors that affect the likelihood of an individual becoming addicted. Create a model that simulates the
likelihood that a given individual will use a given substance. Take into account social influence and characteristic traits
(e.g., social circles, genetics, health issues, income level, and/or any other relevant factors) as well as characteristics
of the drug itself. Demonstrate how your model works by predicting how many students among a class of 300 high

school seniors with varying characteristics will use the following substances: nicotine, marijuana, alcohol, and un-
prescribed opioids."""


problem3 = """Ripples—Develop a robust metric for the impact of substance use. Take into account both financial and non-financial
factors, and use your metric to rank the substances mentioned in question #2."""


