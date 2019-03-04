#problem statements

problem1 = """Darth Vapor—Often containing high doses of nicotine, vaping (inhalation of an aerosol created by vaporizing
a liquid) is hooking a new generation that might otherwise have chosen not to use tobacco products. Build a
mathematical model that predicts the spread of nicotine use due to vaping over the next 10 years. Analyze how the
growth of this new form of nicotine use compares to that of cigarettes."""



problem3 = """Ripples—Develop a robust metric for the impact of substance use. Take into account both financial and non-financial
factors, and use your metric to rank the substances mentioned in question #2."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sys


display = False
if len(sys.argv) > 1:
    display = True
#import sklearn as skl
high_school_drug_use = pd.read_csv("data/NIH-DrugTrends-DataSheet.csv", header = 0)
#print(high_school_drug_use['8th Graders 2015'])
#print(tabulate(high_school_drug_use, headers='keys', tablefmt='psql'))

#graphing libraries

import matplotlib
import matplotlib.pyplot as plt

years = ['2015', '2016', '2017', '2018']
grades = ['8th Graders', '10th Graders', '12th Graders']
colors = {grades[0]: 'b', grades[1]: 'r', grades[2]: 'g'}

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


def format_selection (drug, time_period, grade):
    data_points = [[], []]
    selection = drugs[drug][time_period]
    for year in years:
        data_point = get_gradeyear(year, grade)[selection]
        if data_point == '-':
            continue
        elif data_point[0] == '[' and data_point[-1] == ']':
            y = float(data_point[1:-1])
        else:
            y = float(data_point)
        data_points[0].append(int(year))
        data_points[1].append(y)

    return data_points

def plot_selection(ax, drug, time_period):
    for grade in grades:
        formatted_selection = format_selection(drug, time_period, grade)
        ax.plot(formatted_selection[0], formatted_selection[1], colors[grade], label=grade)
    ax.set_xlabel('years')
    ax.set_ylabel('percent')
    ax.set_title('Usage of {} in {}'.format(drug, time_period))
    ax.set_ylim([0,100])
    ax.legend()

drug = 'Any Vaping'
time_period = 'Past Month'


fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
plot_selection(ax1, 'Any Vaping', 'Past Month')
plot_selection(ax2, 'Cigarettes (any use)', 'Past Month')




# ax.plot(years, data_points)

# fig.savefig('graphs/test.svg')
if display:
    print("displaying")
    plt.show()

