import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import figure

# mean NMI,mean MCC,mean loss,max NMI,max MCC,min loss

results = {
    'AA_type': np.array([]),
    'K': np.array([]),
    'mean NMI': np.array([]),
    'mean MCC': np.array([]),
    'mean loss': np.array([]),
    'max NMI': np.array([]),
    'max MCC': np.array([]),
    'min loss': np.array([]),
}


directory = 'ESS8 results'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath) and not filename == ".DS_Store":
        file = open(filepath,'rb')
        data = pd.read_csv(filepath,encoding = "ISO-8859-1")

        for key in list(results.keys()):
            if not key in ["AA_type","K"]:
                for i in range(len(data[key])):
                    results[key] = np.append(results[key], data[key][i])
        for i in range(len(data["mean NMI"])):
            results["AA_type"] = np.append(results["AA_type"],filename.split("_")[2])
            results["K"] = np.append(results["K"],filename.split("=")[1])




path = "ESS8 plots"
measures = ['mean NMI','mean MCC','mean loss','max NMI','max MCC','min loss']

for measure in measures:

    order_CAA = np.argsort(results["K"][(results["AA_type"] == "CAA")].astype(int))
    CAA_ks = np.array(results["K"][(results["AA_type"] == "CAA")])[order_CAA]
    CAA = np.array(results[measure][results["AA_type"] == "CAA"])[order_CAA]

    order_OAA = np.argsort(results["K"][(results["AA_type"] == "OAA")].astype(int))
    OAA_ks = np.array(results["K"][(results["AA_type"] == "OAA")])[order_OAA]
    OAA = np.array(results[measure][results["AA_type"] == "OAA"])[order_OAA]

    order_RBOAA = np.argsort(results["K"][(results["AA_type"] == "RBOAA")].astype(int))
    RBOAA_ks = np.array(results["K"][(results["AA_type"] == "RBOAA")])[order_RBOAA]
    RBOAA = np.array(results[measure][results["AA_type"] == "RBOAA"])[order_RBOAA]

    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.scatter(CAA_ks,CAA, label = "CAA")
    plt.plot(CAA_ks,CAA, '--')
    plt.scatter(RBOAA_ks,RBOAA, label = "RBOAA")
    plt.plot(RBOAA_ks,RBOAA, '--')
    plt.scatter(OAA_ks,OAA, label = "OAA")
    plt.plot(OAA_ks,OAA, '--')
    plt.xlabel('K', fontsize=14)
    plt.ylabel('{0}'.format(measure), fontsize=14)
    plt.title("{0} over K for AA Methods of ESS8".format(measure), fontsize = 20)
    plt.xticks(OAA_ks, rotation='vertical')
    plt.legend()
    file = ("{0}_ESS8.png".format(measure))
    fig.savefig(os.path.join(path,file),dpi=300)
    plt.close()
    


# AA_types = ["CAA","TSAA","RBOAA","OAA"]

# path = "optimizers plots"

# for AA_type in AA_types:
#     adam_loss = Adam[AA_type]
#     adamNoArmsgrad_loss = Adam_noArsmgrad[AA_type]
#     RMSprop_loss = RMSprop[AA_type]
#     SGD_loss = SGD[AA_type]

#     fig, ax = plt.subplots()
#     fig.set_size_inches(10, 7)

#     ax.boxplot((adam_loss,adamNoArmsgrad_loss,RMSprop_loss,SGD_loss),patch_artist=True,medianprops=dict(color="black"),)
#     plt.title("Loss for {0} over optimizers".format(AA_type), fontsize = 20)
#     plt.xticks(np.arange(6),("","ADAM w. armsgrad","ADAM","RMSprop","SGD",""), fontsize = 15)
#     ax.ticklabel_format(style='plain')
#     file = ("Losses_{0}_optimizers.png".format(AA_type))
#     fig.savefig(os.path.join(path,file),dpi=200)
#     plt.close()