#%%
import pandas as pd
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

results = {"Type": [], "LR": [], "DataSize": [], "Loss": []}
directory = 'LR results'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath) and ".csv" in filepath:
        data = pd.read_csv(filepath)

        AA_type = filename.split("_")[0]
        LR = float(filename.split("_")[1])
        data_size = int(filename.split("_")[2].split(".")[0])
        losses = data["Losses"]

        for loss in losses:
            results["Type"].append(AA_type)
            results["LR"].append(LR)
            results["DataSize"].append(data_size)
            results["Loss"].append(loss)

#print(results["Loss"])

df = pd.DataFrame(results)
#print(df.head)

#df = pd.read_csv('data.csv')
df_caa1 = df[(df["Type"]=="CAA") & (df["DataSize"] == 1000)]
df_caa2 = df[(df["Type"]=="CAA") & (df["DataSize"] == 10000)]
df_caa3 = df[(df["Type"]=="CAA") & (df["DataSize"] == 40000)]
df_tsaa1 = df[(df["Type"]=="TSAA") & (df["DataSize"] == 1000)]
df_tsaa2 = df[(df["Type"]=="TSAA") & (df["DataSize"] == 10000)]
df_tsaa3 = df[(df["Type"]=="TSAA") & (df["DataSize"] == 40000)]
df_oaa1 = df[(df["Type"]=="OAA") & (df["DataSize"] == 1000)]
df_oaa2 = df[(df["Type"]=="OAA") & (df["DataSize"] == 10000)]
df_oaa3 = df[(df["Type"]=="OAA") & (df["DataSize"] == 40000)]
df_rboaa1 = df[(df["Type"]=="RBOAA") & (df["DataSize"] == 1000)]
df_rboaa2 = df[(df["Type"]=="RBOAA") & (df["DataSize"] == 10000)]
df_rboaa3 = df[(df["Type"]=="RBOAA") & (df["DataSize"] == 40000)]


df_caa = df[(df["Type"]=="CAA")]
df_tsaa = df[(df["Type"]=="TSAA")]
df_oaa = df[(df["Type"]=="OAA")]
df_rboaa = df[(df["Type"]=="RBOAA")]

#%%
####  BOXPLOTS ####
sns.set()
"""
fig, axes = plt.subplots(1, 3)
sns.boxplot(x='LR', y='Loss', data=df_caa1, color='#99c2a2', ax=axes[0])
sns.boxplot(x='LR', y='Loss', data=df_caa2, color='#99c2a2', ax=axes[1])
sns.boxplot(x='LR', y='Loss', data=df_caa3, color='#99c2a2', ax=axes[2])
fig.suptitle("CAA", fontsize=16)
axes[0].set_title("1k")
axes[1].set_title("10k")
axes[2].set_title("40k")
#plt.show()

#%%
fig, axes = plt.subplots(1, 3)
sns.boxplot(x='LR', y='Loss', data=df_tsaa1, color='#99c2a2', ax=axes[0])
sns.boxplot(x='LR', y='Loss', data=df_tsaa2, color='#99c2a2', ax=axes[1])
sns.boxplot(x='LR', y='Loss', data=df_tsaa3, color='#99c2a2', ax=axes[2])
fig.suptitle("TSAA", fontsize=16)
axes[0].set_title("1k")
axes[1].set_title("10k")
axes[2].set_title("40k")
#plt.show()

#%%
fig, axes = plt.subplots(1, 3)
sns.boxplot(x='LR', y='Loss', data=df_oaa1, color='#99c2a2', ax=axes[0])
sns.boxplot(x='LR', y='Loss', data=df_oaa2, color='#99c2a2', ax=axes[1])
sns.boxplot(x='LR', y='Loss', data=df_oaa3, color='#99c2a2', ax=axes[2])
fig.suptitle("OAA", fontsize=16)
axes[0].set_title("1k")
axes[1].set_title("10k")
axes[2].set_title("40k")
#plt.show()

#%%
fig, axes = plt.subplots(1, 3)
sns.boxplot(x='LR', y='Loss', data=df_rboaa1, color='#99c2a2', ax=axes[0])
sns.boxplot(x='LR', y='Loss', data=df_rboaa2, color='#99c2a2', ax=axes[1])
sns.boxplot(x='LR', y='Loss', data=df_rboaa3, color='#99c2a2', ax=axes[2])
fig.suptitle("RBOAA", fontsize=16)
axes[0].set_title("1k")
axes[1].set_title("10k")
axes[2].set_title("40k")
#plt.show()
"""
#%%
#### 1-way ANOVA ####
#model type 1-4 and data size 1-3



model_caa1 = ols('Loss ~ LR', data=df_caa1).fit()
model_caa2 = ols('Loss ~ LR', data=df_caa2).fit()
model_caa3 = ols('Loss ~ LR', data=df_caa3).fit()
model_tsaa1 = ols('Loss ~ LR', data=df_tsaa1).fit()
model_tsaa2 = ols('Loss ~ LR', data=df_tsaa2).fit()
model_tsaa3 = ols('Loss ~ LR', data=df_tsaa3).fit()
model_oaa1 = ols('Loss ~ LR', data=df_oaa1).fit()
model_oaa2 = ols('Loss ~ LR', data=df_oaa2).fit()
model_oaa3 = ols('Loss ~ LR', data=df_oaa3).fit()
model_rboaa1 = ols('Loss ~ LR', data=df_rboaa1).fit()
model_rboaa2 = ols('Loss ~ LR', data=df_rboaa2).fit()
model_rboaa3 = ols('Loss ~ LR', data=df_rboaa3).fit()

model_names = ["model_caa1", "model_caa2", "model_caa3", "model_tsaa1", "model_tsaa2", "model_tsaa3", "model_oaa1", "model_oaa2", 
          "model_oaa3", "model_rboaa1", "model_rboaa2", "model_rboaa3"]
linear_models = [model_caa1, model_caa2, model_caa3, model_tsaa1, model_tsaa2, model_tsaa3, model_oaa1, model_oaa2, 
          model_oaa3, model_rboaa1, model_rboaa2, model_rboaa3]

anova_tables = {}
for i in range(len(model_names)):
    anova_tables[model_names[i]] = sm.stats.anova_lm(linear_models[i], typ=1)
    print("\n \n ####", model_names[i])
    print(anova_tables[model_names[i]])
 


### flere plots
sns.scatterplot(x='LR', y='Loss', data=df_caa2, color='#99c2a2')
plt.title("CAA", fontsize=16)
plt.savefig("LR results/scat_CAA.png")
plt.show()

sns.scatterplot(x='LR', y='Loss', data=df_tsaa2, color='#99c2a2')
plt.title("TSAA", fontsize=16)
plt.savefig("LR results/scat_TSAA.png")
plt.show()

sns.scatterplot(x='LR', y='Loss', data=df_oaa2, color='#99c2a2')
plt.title("OAA", fontsize=16)
plt.savefig("LR results/scat_OAA.png")
plt.show()

sns.scatterplot(x='LR', y='Loss', data=df_rboaa2, color='#99c2a2')
plt.title("RBOAA", fontsize=16)
plt.savefig("LR results/scat_RBOAA.png")
plt.show()


model_caa = ols('Loss ~ LR + DataSize + LR:DataSize', data=df_caa).fit()
model_tsaa = ols('Loss ~ LR + DataSize + LR:DataSize', data=df_tsaa).fit()
model_oaa = ols('Loss ~ LR + DataSize + LR:DataSize', data=df_oaa).fit()
model_rboaa = ols('Loss ~ LR + DataSize + LR:DataSize', data=df_rboaa).fit()

m1 = sm.stats.anova_lm(model_caa, typ=2)
m2= sm.stats.anova_lm(model_tsaa, typ=2)
m3= sm.stats.anova_lm(model_oaa, typ=2)
m4= sm.stats.anova_lm(model_rboaa, typ=2)
print("\n\n\n\n\n YEEAAAH BUDDDY")
print("\n \n &&&&&& CAA\n",m1)
print("\n \n &&&&&& TSAA\n",m2)
print("\n \n &&&&&& OAA\n",m3)
print("\n \n &&&&&& RBOAA\n",m4)
