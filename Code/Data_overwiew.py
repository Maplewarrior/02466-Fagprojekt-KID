import pandas as pd
import matplotlib.pyplot as plt
import math
data_all=pd.read_csv(r"C:\Users\Andre\OneDrive - Danmarks Tekniske Universitet\Bachelor project\Schmidt_et_al_2021_Latent_profile_analysis_of_human_values_SUPPL\VB_LPA\Data\ESS8_data.csv")
keys=["SD1","PO1","UN1","AC1","SC1","ST1","CO1","UN2","TR1","HD1","SD2","BE1","AC2","SC2","ST2","CO2","PO2","BE2","UN3","TR2","HD2"]
data=data_all[keys]


#n,m=data.shape

#plot data
fig,axs=plt.subplots(5,5,figsize=(15, 20), dpi=100)
for n,key in enumerate(keys):
    print(n)
    x=[1,2,3,4,5,6]
    bars=data[key].value_counts()[x]
    axs[math.floor(n/5),n%5].bar(x,bars[x])
    axs[math.floor(n/5),n%5].set_title(key)

plt.show()