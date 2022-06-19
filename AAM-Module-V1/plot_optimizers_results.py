import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import figure
import seaborn as sns

Adam = pd.read_csv("optimizers results/optimizers_test_ADAM.csv")
Adam_noArsmgrad = pd.read_csv("optimizers results/optimizers_test_ADAM_noArmsgrad.csv")
RMSprop = pd.read_csv("optimizers results/optimizers_test_RMSprop.csv")
SGD = pd.read_csv("optimizers results/optimizers_test_SGD.csv")

AA_types = ["CAA","TSAA","RBOAA","OAA"]

path = "optimizers plots"

for AA_type in AA_types:
    adam_loss = Adam[AA_type]
    adamNoArmsgrad_loss = Adam_noArsmgrad[AA_type]
    RMSprop_loss = RMSprop[AA_type]
    SGD_loss = SGD[AA_type]

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)
    #sns.boxplot((adam_loss,adamNoArmsgrad_loss,RMSprop_loss,SGD_loss),)
    box = ax.boxplot((adam_loss,adamNoArmsgrad_loss,RMSprop_loss,SGD_loss),patch_artist=True, medianprops=dict(color="black"))
    
    colors = ["#cce6ff","#80bfff","#1e90ff","#0073e6"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    #ax.ticklabel_format(style='plain')
    plt.title("Loss for {0} over optimizers".format(AA_type), fontsize = 20)
    plt.ylabel('Loss', fontsize=15)
    plt.xticks(np.arange(6),("","AMSGrad","ADAM","RMSprop","SGD",""), fontsize = 15)
    ax.set_facecolor('xkcd:salmon')

    file = ("Losses_{0}_optimizers.png".format(AA_type))
    fig.savefig(os.path.join(path,file),dpi=200)
    plt.close()