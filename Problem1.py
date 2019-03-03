import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pandas2ri.activate()

# print(r['load']("data/NSDUH_2002_2016.RData"))

data = pd.read_csv('data/E-CigaretteUseAmongYouth.csv')

E_cigarette_use_total_percentage_last_30_days = data.iloc[16].str[:4][[1,3,5,7,9]].astype('float').add(
    data.iloc[17].str[:4][[1,3,5,7,9]].astype('float').add(data.iloc[18].str[:4][[1,3,5,7,9]].astype('float').add(
    data.iloc[19].str[:4][[1,3,5,7,9]].astype('float'))))

Cigarette_use_total_percentage_last_30_days = data.iloc[17].str[:4][[1,3,5,7,9]].astype('float').add(
    data.iloc[19].str[:4][[1,3,5,7,9]].astype('float').add(data.iloc[20].str[:4][[1,3,5,7,9]].astype('float').add(
    data.iloc[21].str[:4][[1,3,5,7,9]].astype('float'))))

print(Cigarette_use_total_percentage_last_30_days)

