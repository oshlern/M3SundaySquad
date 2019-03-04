import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

display = False
if len(sys.argv) > 1:
    display = True
save = True

# pandas2ri.activate()

# print(r['load']("data/NSDUH_2002_2016.RData"))

data = pd.read_csv('data/E-CigaretteUseAmongYouth.csv')

E_cigarette_use_total_percentage_last_30_days = data.iloc[16].str[:4][[1,3,5,7,9]].astype('float').add(
    data.iloc[17].str[:4][[1,3,5,7,9]].astype('float').add(data.iloc[18].str[:4][[1,3,5,7,9]].astype('float').add(
    data.iloc[19].str[:4][[1,3,5,7,9]].astype('float'))))

Cigarette_use_total_percentage_last_30_days = data.iloc[17].str[:4][[1,3,5,7,9]].astype('float').add(
    data.iloc[19].str[:4][[1,3,5,7,9]].astype('float').add(data.iloc[20].str[:4][[1,3,5,7,9]].astype('float').add(
    data.iloc[21].str[:4][[1,3,5,7,9]].astype('float'))))

Cigarette_data = pd.read_csv('data/Cigarette_data.csv', header = None)

Cigarettte_years = list(Cigarette_data.iloc[0][1:].astype('int'))
print(Cigarettte_years)
tenth = list(Cigarette_data.iloc[2][1:].astype('float'))
print("tenth", tenth)
twelth = list(Cigarette_data.iloc[3][1:].astype('float'))

years = [2011, 2012, 2013, 2014, 2015]

#print(Cigarette_use_total_percentage_last_30_days)

fig = plt.figure()
ax = fig.add_subplot(131)
ax.plot(years, E_cigarette_use_total_percentage_last_30_days, color = 'b', label = "E-cigarettes")
ax.plot(years, Cigarette_use_total_percentage_last_30_days, color = 'r', label = "Cigarettes")

ax.set_xlabel('Year')
ax.set_ylabel('Percent')
ax.set_title('Percent Highschoolers that Regularly use E-cigaretts')
ax.legend()


ax2 = fig.add_subplot(132)
ax2.plot(Cigarettte_years, tenth, color = 'g', label="10th")
ax2.plot(Cigarettte_years, twelth, color = 'y', label="12th")

ax2.set_xlabel('Year')
ax2.set_ylabel('Percent')
ax2.set_title('Percent used by Highschoolers')
ax2.legend()


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
def reformat(mylist):
    reformatted = []
    for x in mylist:
        reformatted.append([x])
    return reformatted
#print(Cigarette_use_total_percentage_last_30_days)
ax3 = fig.add_subplot(133)
cigs = linear_model.LinearRegression()
cigs.fit(reformat(years), reformat(Cigarette_use_total_percentage_last_30_days))
pred = cigs.predict(reformat(years))
ax3.plot(years, pred, color = 'g')
ecigs = linear_model.LinearRegression()
ecigs.fit(reformat(years), reformat(E_cigarette_use_total_percentage_last_30_days))
pred = ecigs.predict(reformat(years))
ax3.plot(years, pred, color = 'black')

if save:
    fig.savefig('graphs/problem1.svg')
print(ecigs.predict([[2029]]))

if display:
    plt.show()
