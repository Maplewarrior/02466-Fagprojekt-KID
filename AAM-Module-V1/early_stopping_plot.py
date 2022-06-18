import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import figure
import matplotlib
import seaborn as sns

RBOAA_ES = []
RBOAA_NES = []

OAA_ES = []
OAA_NES = []

CAA_ES = []
CAA_NES = []

directory = 'early stopping results'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath) and not filename == ".DS_Store":
        file = open(filepath,'rb')
        data = pd.read_csv(filepath,encoding = "ISO-8859-1")

        if data["AA type"][0] == "RBOAA":
            if data["early_stopping"][0] == False:
                for value in data["losses"]:
                    RBOAA_NES.append(value)
            else:
                for value in data["losses"]:
                    RBOAA_ES.append(value)
        
        elif data["AA type"][0] == "OAA":
            if data["early_stopping"][0] == False:
                for value in data["losses"]:
                    OAA_NES.append(value)
            else:
                for value in data["losses"]:
                    OAA_ES.append(value)
        
        elif data["AA type"][0] == "CAA":
            if data["early_stopping"][0] == False:
                for value in data["losses"]:
                    CAA_NES.append(value)
            else:
                for value in data["losses"]:
                    CAA_ES.append(value)


path = "early stopping plots"

fig, ax = plt.subplots()
fig.set_size_inches(10, 7)

ax.boxplot((RBOAA_NES,RBOAA_ES), patch_artist=True, boxprops=dict(facecolor="#cce6ff", color="#cce6ff"),medianprops=dict(color="black"),)
plt.title("Effect of Early Stopping on Loss of RBOAA", fontsize = 20)
plt.xticks(np.arange(4),("","No Early Stopping","Early Stopping",""), fontsize = 15)
ax.set_ylabel('Loss', fontsize=15)
file = ("ES_RBOAA.png")
fig.savefig(os.path.join(path,file),dpi=200)
plt.close()

fig, ax = plt.subplots()
fig.set_size_inches(10, 7)

ax.boxplot((OAA_NES,OAA_ES), patch_artist=True, boxprops=dict(facecolor="#cce6ff", color="#cce6ff"),medianprops=dict(color="black"),)
plt.title("Effect of Early Stopping on Loss of OAA", fontsize = 20)
plt.xticks(np.arange(4),("","No Early Stopping","Early Stopping",""), fontsize = 15)
ax.set_ylabel('Loss', fontsize=15)
file = ("ES_OAA.png")
fig.savefig(os.path.join(path,file),dpi=200)
plt.close()

fig, ax = plt.subplots()
fig.set_size_inches(10, 7)

ax.boxplot((CAA_NES,CAA_ES), patch_artist=True, boxprops=dict(facecolor="#cce6ff", color="#cce6ff"),medianprops=dict(color="black"),)
plt.title("Effect of Early Stopping on Loss of CAA", fontsize = 20)
plt.xticks(np.arange(4),("","No Early Stopping","Early Stopping",""), fontsize = 15)
ax.set_ylabel('Loss', fontsize=15)
file = ("ES_CAA.png")
fig.savefig(os.path.join(path,file),dpi=200)
plt.close()