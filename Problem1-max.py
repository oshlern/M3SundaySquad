

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sys

def load_csv(path, year, years):
    bad_data = pd.read_csv(path).reindex(years[year].keys(), axis=1).fillna('*')
    data = {}
    for question in years[year]:
        data[question] = []
        bad_data[question]
        # try:
        for entry in bad_data[question]:
            x1 = years[year]
            x2 = x1[question]
            # print(entry, type(entry), entry == float('nan'), years[year][question])
            x3 = x2[entry]
            data[question].append(x3)
            #[years[year][question][entry] for entry in bad_data[question]]
        # except KeyError:
        #     print("question {}, year {}".format(question, year))
    # data.reindex(years[year].keys(), axis=1).fillna('*')
    # data[data.columns[0]] = data[data.columns[0]].apply(lambda x: 0 if x == "**" else int(x)+8)
    # seen = []
    # for i, x in enumerate(data[data.columns[1]]):
    #     # if type(x) != str:
    #     #     print(i, x)
    #     if not x in seen:
    #         seen.append(x)
    return data

years = {
    '2017': {
        'Q1': {'*': '<Missing>', '**': '<Missing>', '01': 9, 1: 9, '02': 10, 2: 10, '03': 11, 3: 11, '04': 12, 4: 12, '05': 13, 5: 13, '06': 14, 6: 14, '07': 15, 7: 15, '08': 16, 8: 16, '09': 17, 9: 17, '10': 18, 10: 18, '11': 19, 11: 19}, # How old are you?
        'Q2': {'*': '<Missing>', '1': 'Male', 1: 'Male', '2': 'Female', 2: 'Female'}, # What is your sex?
        'Q3': {'*': '<Missing>', '1': 6, 1: 6, '2': 7, 2: 7, '3': 8, 3: 8, '4': 9, 4: 9, '5': 10, 5: 10, '6': 11, 6: 11, '7': 12, 7: 12, '8': '<Missing>', 8: '<Missing>'}, # What grade are you in?
        'Q7': {'*': '<Missing>', '1': True, 1: True, '2': False, 2: False}, # Have you ever tried cigarette smoking, even one or two puffs?
        'Q28': {'*': '<Missing>', '1': True, 1: True, '2': False, 2: False} # Have you ever used an e-cigarette, even once or twice?
    },
    '2016': {
        'Q1': {'*': '<Missing>', '**': '<Missing>', '01': 9, 1: 9, '02': 10, 2: 10, '03': 11, 3: 11, '04': 12, 4: 12, '05': 13, 5: 13, '06': 14, 6: 14, '07': 15, 7: 15, '08': 16, 8: 16, '09': 17, 9: 17, '10': 18, 10: 18, '11': 19, 11: 19}, # How old are you?
        'Q2': {'*': '<Missing>', '1': 'Male', 1: 'Male', '2': 'Female', 2: 'Female'}, # What is your sex?
        'Q3': {'*': '<Missing>', '1': 6, 1: 6, '2': 7, 2: 7, '3': 8, 3: 8, '4': 9, 4: 9, '5': 10, 5: 10, '6': 11, 6: 11, '7': 12, 7: 12, '8': '<Missing>', 8: '<Missing>'}, # What grade are you in?
        'Q7': {'*': '<Missing>', '1': True, 1: True, '2': False, 2: False}, # Have you ever tried cigarette smoking, even one or two puffs?
        'Q26': {'*': '<Missing>', '1': True, 1: True, '2': False, 2: False} # Have you ever used an e-cigarette, even once or twice?
    },
    '2015': {
        'q1': {'*': '<Missing>', '**': '<Missing>', '01': 9, 1: 9, '02': 10, 2: 10, '03': 11, 3: 11, '04': 12, 4: 12, '05': 13, 5: 13, '06': 14, 6: 14, '07': 15, 7: 15, '08': 16, 8: 16, '09': 17, 9: 17, '10': 18, 10: 18, '11': 19, 11: 19}, # How old are you?
        'q2': {'*': '<Missing>', '1': 'Male', 1: 'Male', '2': 'Female', 2: 'Female'}, # What is your sex?
        'q3': {'*': '<Missing>', '1': 6, 1: 6, '2': 7, 2: 7, '3': 8, 3: 8, '4': 9, 4: 9, '5': 10, 5: 10, '6': 11, 6: 11, '7': 12, 7: 12, '8': '<Missing>', 8: '<Missing>'}, # What grade are you in?
        'q6': {'*': '<Missing>', 'E': '<Missing>', '1': True, 1: True, '2': False, 2: False}, # Have you ever tried cigarette smoking, even one or two puffs?
        'q28': {'*': '<Missing>', 'E': '<Missing>', '1': True, 1: True, '2': False, 2: False} # Have you ever used an e-cigarette, even once or twice?
    },
    '2014': {
        'qn1': {'*': '<Missing>', '**': '<Missing>', 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18, 11: 19}, # Age
        'qn2': {'*': '<Missing>', 1: 'Male', 2: 'Female'}, # Sex
        'qn3': {'*': '<Missing>', 1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 11, 7: 12, 8: '<Missing>'}, # Grade
        'qn7': {'*': '<Missing>', 'E': '<Missing>', 1: True, 2: False}, # Tried cigarette smkg, even 1 or 2 puffs
        'qn31': {'*': False, 'E': '<Missing>', 1: True, 2: False} # Ever tried electronic cigarettes
    },
    '2013': {
        'qn1': {'*': '<Missing>', '**': '<Missing>', 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18, 11: 19}, # Age
        'qn2': {'*': '<Missing>', 1: 'Male', 2: 'Female'}, # Sex
        'qn3': {'*': '<Missing>', 1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 11, 7: 12, 8: '<Missing>'}, # Grade
        'qn9': {'*': '<Missing>', 'E': '<Missing>', 1: True, 2: False}, # Tried cigarette smkg, even 1 or 2 puffs
        'qn36i': {'*': False, 'E': '<Missing>', 1: True} # EVER TRIED: e-cigt (e.g. Ruyan)
    },
    '2012': {
        'qn1': {'*': '<Missing>', '**': '<Missing>', 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18, 11: 19}, # Age
        'qn2': {'*': '<Missing>', 1: 'Female', 2: 'Male'}, # Sex
        'qn3': {'*': '<Missing>', 1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 11, 7: 12, 8: '<Missing>'}, # Grade
        'qn7': {'*': '<Missing>', 'E': '<Missing>', 1: True, 2: False}, # Tried cigarette smkg, even 1 or 2 puffs
        'qn37g': {'*': False, 'I': '<Missing>', 1: True} # EVER TRIED: e-cigt (e.g. Ruyan)
    },
    '2011': {
        'qn1': {'*': '<Missing>', '**': '<Missing>', 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18, 11: 19}, # Age
        'qn2': {'*': '<Missing>', 1: 'Female', 2: 'Male'}, # Sex
        'qn3': {'*': '<Missing>', 1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 11, 7: 12, 8: '<Missing>'}, # Grade
        'qn7': {'*': '<Missing>', 1: True, 2: False}, # Tried cigarette smkg, even 1 or 2 puffs
        'qn36h': {'*': False, 1: True} # EVER TRIED: e-cigt (e.g. Ruyan)
    },
}

files = {
    '2017' : 'data/NYTS/nyts2017.csv',
    '2016' : 'data/NYTS/nyts2016.csv',
    '2015' : 'data/NYTS/nyts2015.csv',
    '2014' : 'data/NYTS/nyts2014.csv',
    '2013' : 'data/NYTS/nyts2013.csv',
    '2012' : 'data/NYTS/nyts2012.csv',
    '2011' : 'data/NYTS/nyts2011.csv'
}

data = {year: load_csv(files[year], year, years) for year in files}

# data {'2017': {'Q1': [entries]}}

