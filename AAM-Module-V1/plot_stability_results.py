import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import figure

data = pd.read_csv("full_stability_dataset.csv")
data.sort_values(by=['sigma'], inplace=True)

path = os.getcwd()
foldername = "stability plots"
path = os.path.join(path, foldername)

data["sigma"] = np.log(1+np.exp(data["sigma"]))

CAA = data[data["AA_type"] == "CAA"]
TSAA = data[data["AA_type"] == "TSAA"]
OAA = data[data["AA_type"] == "OAA"]
RBOAA = data[data["AA_type"] == "RBOAA"]

fig = plt.gcf()
fig.set_size_inches(10, 7)

plt.scatter(CAA["sigma"],CAA["A_SDM"], label = "CAA")
plt.plot(CAA["sigma"],CAA["A_SDM"], '--')

plt.scatter(TSAA["sigma"],TSAA["A_SDM"], label = "TSAA")
plt.plot(TSAA["sigma"],TSAA["A_SDM"], '--')

plt.scatter(OAA["sigma"],OAA["A_SDM"], label = "OAA")
plt.plot(OAA["sigma"],OAA["A_SDM"], '--')

plt.scatter(RBOAA["sigma"],RBOAA["A_SDM"], label = "RBOAA")
plt.plot(RBOAA["sigma"],RBOAA["A_SDM"], '--')

plt.xlabel('sigma', fontsize=14)
plt.ylabel('Mean Squared Distance', fontsize=14)
plt.title("MSD of A over 30 itterations", fontsize = 20)
plt.legend()
file = ("A MSD over Sigma.png")
fig.savefig(os.path.join(path,file),dpi=100)
plt.close()


fig = plt.gcf()
fig.set_size_inches(10, 7)

plt.scatter(CAA["sigma"],CAA["Z_SDM"], label = "CAA")
plt.plot(CAA["sigma"],CAA["Z_SDM"], '--')

plt.scatter(TSAA["sigma"],TSAA["Z_SDM"], label = "TSAA")
plt.plot(TSAA["sigma"],TSAA["Z_SDM"], '--')

plt.scatter(OAA["sigma"],OAA["Z_SDM"], label = "OAA")
plt.plot(OAA["sigma"],OAA["Z_SDM"], '--')

plt.scatter(RBOAA["sigma"],RBOAA["Z_SDM"], label = "RBOAA")
plt.plot(RBOAA["sigma"],RBOAA["Z_SDM"], '--')

plt.xlabel('sigma', fontsize=14)
plt.ylabel('Mean Squared Distance', fontsize=14)
plt.title("MSD of Z over 30 itterations", fontsize = 20)
plt.legend()
file = ("Z MSD over Sigma.png")
fig.savefig(os.path.join(path,file),dpi=200)
plt.close()