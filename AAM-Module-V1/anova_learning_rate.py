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

df = pd.DataFrame(results)

#For each model
df_caa = df[(df["Type"]=="CAA")]
df_tsaa = df[(df["Type"]=="TSAA")]
df_oaa = df[(df["Type"]=="OAA")]
df_rboaa = df[(df["Type"]=="RBOAA")]

# For each model and datasize
df_caa1 = df[(df["Type"]=="CAA") & (df["DataSize"] == 1000)]
df_caa2 = df[(df["Type"]=="CAA") & (df["DataSize"] == 5000)]
df_caa3 = df[(df["Type"]=="CAA") & (df["DataSize"] == 10000)]
df_tsaa1 = df[(df["Type"]=="TSAA") & (df["DataSize"] == 1000)]
df_tsaa2 = df[(df["Type"]=="TSAA") & (df["DataSize"] == 5000)]
df_tsaa3 = df[(df["Type"]=="TSAA") & (df["DataSize"] == 10000)]
df_oaa1 = df[(df["Type"]=="OAA") & (df["DataSize"] == 1000)]
df_oaa2 = df[(df["Type"]=="OAA") & (df["DataSize"] == 5000)]
df_oaa3 = df[(df["Type"]=="OAA") & (df["DataSize"] == 10000)]
df_rboaa1 = df[(df["Type"]=="RBOAA") & (df["DataSize"] == 1000)]
df_rboaa2 = df[(df["Type"]=="RBOAA") & (df["DataSize"] == 5000)]
df_rboaa3 = df[(df["Type"]=="RBOAA") & (df["DataSize"] == 10000)]

# For 10k sets
df_caa3_1 = df_caa3[df_caa3['LR']==0.1]
df_caa3_2 = df_caa3[df_caa3['LR']==0.05]
df_caa3_3 = df_caa3[df_caa3['LR']==0.01]
df_caa3_4 = df_caa3[df_caa3['LR']==0.005]
df_caa3_5 = df_caa3[df_caa3['LR']==0.001]

df_oaa3_1 = df_oaa3[df_oaa3['LR']==0.1]
df_oaa3_2 = df_oaa3[df_oaa3['LR']==0.05]
df_oaa3_3 = df_oaa3[df_oaa3['LR']==0.01]
df_oaa3_4 = df_oaa3[df_oaa3['LR']==0.005]
df_oaa3_5 = df_oaa3[df_oaa3['LR']==0.001]

df_rboaa3_1 = df_rboaa3[df_rboaa3['LR']==0.1]
df_rboaa3_2 = df_rboaa3[df_rboaa3['LR']==0.05]
df_rboaa3_3 = df_rboaa3[df_rboaa3['LR']==0.01]
df_rboaa3_4 = df_rboaa3[df_rboaa3['LR']==0.005]
df_rboaa3_5 = df_rboaa3[df_rboaa3['LR']==0.001]
#%%
####  BOXPLOTS ####
sns.set()
sns.set_style("white")
#sns.set_palette(sns.color_palette(["#cce6ff","#80bfff","#1e90ff","#0073e6", "#0059b3"]))
"""
fig, axes = plt.subplots(1,3)
fig.set_size_inches(10, 7)
sns.boxplot(x='LR', y='Loss', data=df_caa1, ax=axes[0])
sns.boxplot(x='LR', y='Loss', data=df_caa2, ax=axes[1])
sns.boxplot(x='LR', y='Loss', data=df_caa3, ax=axes[2])
fig.suptitle("CAA", fontsize=16)
axes[0].set_title("1k")
axes[1].set_title("5k")
axes[2].set_title("10k")
fig.subplots_adjust(wspace=0.4)
fig.savefig("LR results/box_caa.png",bbox_inches='tight', dpi=200)
#plt.show()

#sns.set_palette("Greens", n_colors=5)
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 7)
sns.boxplot(x='LR', y='Loss', data=df_tsaa1, ax=axes[0])
sns.boxplot(x='LR', y='Loss', data=df_tsaa2, ax=axes[1])
sns.boxplot(x='LR', y='Loss', data=df_tsaa3, ax=axes[2]) #color='#99c2a2'
fig.suptitle("TSAA", fontsize=16)
axes[0].set_title("1k")
axes[1].set_title("5k")
axes[2].set_title("10k")
fig.subplots_adjust(wspace=0.4)
plt.savefig("LR results/box_tsaa.png", bbox_inches='tight', dpi=200)
#plt.show()

#sns.set_palette("Greens", n_colors=5)
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 7)
sns.boxplot(x='LR', y='Loss', data=df_oaa1, ax=axes[0])
sns.boxplot(x='LR', y='Loss', data=df_oaa2, ax=axes[1])
sns.boxplot(x='LR', y='Loss', data=df_oaa3, ax=axes[2])
fig.suptitle("OAA", fontsize=16)
axes[0].set_title("1k")
axes[1].set_title("5k")
axes[2].set_title("10k")
fig.subplots_adjust(wspace=0.4)
plt.savefig("LR results/box_oaa.png", bbox_inches='tight',dpi=200)
#plt.show()

fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 7)
sns.boxplot(x='LR', y='Loss', data=df_rboaa1, ax=axes[0])
sns.boxplot(x='LR', y='Loss', data=df_rboaa2, ax=axes[1])
sns.boxplot(x='LR', y='Loss', data=df_rboaa3, ax=axes[2])
fig.suptitle("RBOAA", fontsize=16)
axes[0].set_title("1k")
axes[1].set_title("5k")
axes[2].set_title("10k")
fig.subplots_adjust(wspace=0.5)
plt.savefig("LR results/box_rboaa.png",bbox_inches='tight', dpi=200)
plt.show()
"""

#%%
#### 1-way ANOVA ####
#Data size 1-3
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
#linear_models = [model_caa1, model_caa2, model_caa3, model_oaa1, model_oaa2, model_oaa3, model_rboaa1, model_rboaa2, model_rboaa3]

anova_tables = {}
for i in range(len(model_names)):
    anova_tables[model_names[i]] = sm.stats.anova_lm(linear_models[i], typ=1)
    #print("\n \n ####", model_names[i])
    #print(anova_tables[model_names[i]])

#Create model residuals
influence_caa1 = model_caa1.get_influence()
influence_caa2 = model_caa2.get_influence()
influence_caa3 = model_caa3.get_influence()
influence_oaa1 = model_oaa1.get_influence()
influence_oaa2 = model_oaa2.get_influence()
influence_oaa3 = model_oaa3.get_influence()
influence_rboaa1 = model_rboaa1.get_influence()
influence_rboaa2 = model_rboaa2.get_influence()
influence_rboaa3 = model_rboaa3.get_influence()

# Obtain standardized residuals
standardized_residuals_caa1 = influence_caa1.resid_studentized_internal
standardized_residuals_caa2 = influence_caa2.resid_studentized_internal
standardized_residuals_caa3 = influence_caa3.resid_studentized_internal
standardized_residuals_oaa1 = influence_oaa1.resid_studentized_internal
standardized_residuals_oaa2 = influence_oaa2.resid_studentized_internal
standardized_residuals_oaa3 = influence_oaa3.resid_studentized_internal
standardized_residuals_rboaa1 = influence_rboaa1.resid_studentized_internal
standardized_residuals_rboaa2 = influence_rboaa2.resid_studentized_internal
standardized_residuals_rboaa3 = influence_rboaa3.resid_studentized_internal

#%%
#import bioinfokit.analys
from scipy import stats
import numpy as np
#Check for normality within groups
print("SHAPIRO \nCAA W-test stat, p-value", stats.shapiro(model_caa3.resid))
print("OAA W-test stat, p-value", stats.shapiro(model_oaa3.resid))
print("RBOAA W-test stat, p-value", stats.shapiro(model_rboaa3.resid))

print("\n ktest:\nRBOAA W-test stat, p-value", stats.kstest(model_rboaa3.resid, 'norm'))
print("\n\n")

print("SHAPIRO \nCAA W-test stat, p-value", stats.shapiro(standardized_residuals_caa3))
print("OAA W-test stat, p-value", stats.shapiro(standardized_residuals_oaa3))
print("RBOAA W-test stat, p-value", stats.shapiro(standardized_residuals_rboaa3))

fig, axes = plt.subplots(1,3)
fig.set_size_inches(10, 7)
sns.histplot(df_caa, x='Loss', bins='auto', ax=axes[0]) 
sns.histplot(df_oaa, x='Loss', bins='auto', ax=axes[1]) 
sns.histplot(df_rboaa, x='Loss', bins='auto', ax=axes[2]) 
fig.suptitle("Histograms of Models Data", fontsize=16)
axes[0].set_title("CAA")
axes[1].set_title("OAA")
axes[2].set_title("RBOAA")
fig.subplots_adjust(wspace=0.4)
fig.savefig("LR results/hists_models_10k.png",bbox_inches='tight', dpi=200)
plt.show()


log_caa_loss = np.log(df_caa['Loss'])
sns.histplot(log_caa_loss,bins='auto')
plt.show()
log_oaa_loss = np.log(df_oaa['Loss'])
sns.histplot(log_oaa_loss,bins='auto')
plt.show()
log_rboaa_loss = np.log(df_rboaa['Loss'])
sns.histplot(log_rboaa_loss,bins='auto')
plt.show()
sq_oaa_loss = np.sqrt(df_oaa['Loss'])
sns.histplot(sq_oaa_loss,bins='auto')
plt.show()
box_oaa_loss = stats.boxcox(df_oaa['Loss'])
sns.histplot(box_oaa_loss,bins='auto')
plt.show()

"""sns.set_palette(sns.color_palette(["#1e90ff","#0073e6"]))
fig, axes = plt.subplots(2,3)
fig.set_size_inches(10, 7)
sns.histplot(df_caa3_1, x='Loss',bins='auto', ax=axes[0,0]) 
sns.histplot(df_caa3_2, x='Loss',bins='auto', ax=axes[0,1]) 
sns.histplot(df_caa3_3, x='Loss',bins='auto', ax=axes[0,2])
sns.histplot(df_caa3_4, x='Loss',bins='auto', ax=axes[1,0])
sns.histplot(df_caa3_5, x='Loss',bins='auto', ax=axes[1,1]) 
fig.suptitle("CAA 10k", fontsize=16)
axes[0,0].set_title("0.1")
axes[0,1].set_title("0.05")
axes[0,2].set_title("0.01")
axes[1,0].set_title("0.005")
axes[1,1].set_title("0.001")
fig.subplots_adjust(wspace=0.4, hspace=0.4)
fig.savefig("LR results/hists_caa_10k_all_LR.png",bbox_inches='tight', dpi=200)
plt.show()

fig, axes = plt.subplots(2,3)
fig.set_size_inches(10, 7)
sns.histplot(df_oaa3_1, x='Loss',bins='auto', ax=axes[0,0]) 
sns.histplot(df_oaa3_2, x='Loss',bins='auto', ax=axes[0,1]) 
sns.histplot(df_oaa3_3, x='Loss',bins='auto', ax=axes[0,2])
sns.histplot(df_oaa3_4, x='Loss',bins='auto', ax=axes[1,0])
sns.histplot(df_oaa3_5, x='Loss',bins='auto', ax=axes[1,1]) 
fig.suptitle("OAA 10k", fontsize=16)
axes[0,0].set_title("0.1")
axes[0,1].set_title("0.05")
axes[0,2].set_title("0.01")
axes[1,0].set_title("0.005")
axes[1,1].set_title("0.001")
fig.subplots_adjust(wspace=0.4, hspace=0.4)
fig.savefig("LR results/hists_oaa_10k_all_LR.png",bbox_inches='tight', dpi=200)
plt.show()"""

fig, axes = plt.subplots(2,3)
fig.set_size_inches(10, 7)
sns.histplot(df_rboaa3_1, x='Loss',bins='auto', ax=axes[0,0])
sns.histplot(df_rboaa3_2, x='Loss',bins='auto', ax=axes[0,1]) 
sns.histplot(df_rboaa3_3, x='Loss',bins='auto', ax=axes[0,2])
sns.histplot(df_rboaa3_4, x='Loss',bins='auto', ax=axes[1,0])
sns.histplot(df_rboaa3_5, x='Loss',bins='auto', ax=axes[1,1]) 
fig.suptitle("RBOAA 10k", fontsize=16)
axes[0,0].set_title("0.1")
axes[0,1].set_title("0.05")
axes[0,2].set_title("0.01")
axes[1,0].set_title("0.005")
axes[1,1].set_title("0.001")
fig.subplots_adjust(wspace=0.4, hspace=0.4)
fig.savefig("LR results/hists_rboaa_10k_all_LR.png",bbox_inches='tight', dpi=200)
plt.show()

print("CAA: ",stats.kruskal(df_caa3_1['Loss'], df_caa3_2['Loss'], df_caa3_3['Loss'], df_caa3_4['Loss'],df_caa3_5['Loss']))
print("OAA: ",stats.kruskal(df_oaa3_1['Loss'], df_oaa3_2['Loss'], df_oaa3_3['Loss'], df_oaa3_4['Loss'],df_oaa3_5['Loss']))
print("RBOAA: ",stats.kruskal(df_rboaa3_1['Loss'], df_rboaa3_2['Loss'], df_rboaa3_3['Loss'], df_rboaa3_4['Loss'],df_rboaa3_5['Loss']))




"""fig, axes = plt.subplots(1,3)
fig.set_size_inches(10, 7)
sm.qqplot(standardized_residuals_caa3, line='45', ax=axes[0])
sm.qqplot(standardized_residuals_oaa3, line='45', ax=axes[1])
sm.qqplot(standardized_residuals_rboaa3, line='45', ax=axes[2]) 
fig.suptitle("10k", fontsize=16)
axes[0].set_title("caa")
axes[1].set_title("oaa")
axes[2].set_title("rboaa")
fig.subplots_adjust(wspace=0.4)
fig.savefig("LR results/qq_models.png",bbox_inches='tight', dpi=200)
plt.show()"""

fig, axes = plt.subplots(1,3)
fig.set_size_inches(10, 7)
sm.qqplot(standardized_residuals_rboaa1, line='45', ax=axes[0])
sm.qqplot(standardized_residuals_rboaa2, line='45', ax=axes[1])
sm.qqplot(standardized_residuals_rboaa3, line='45', ax=axes[2]) 
fig.suptitle("RBOAA", fontsize=16)
axes[0].set_title("1k")
axes[1].set_title("5k")
axes[2].set_title("10k")
fig.subplots_adjust(wspace=0.4)
fig.savefig("LR results/qq_rboaa.png",bbox_inches='tight', dpi=200)
plt.show()
