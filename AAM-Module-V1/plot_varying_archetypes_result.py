import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import figure

data = pd.read_csv("full_varying_archetypes_dataset.csv")

path = os.getcwd()
foldername = "varying archetype plots"
path = os.path.join(path, foldername)

for i in [3,5,7]:
    CAA = data[(data["synthetic_k"] == i) & (data["AA_type"] == "CAA")]
    TSAA = data[(data["synthetic_k"] == i) & (data["AA_type"] == "TSAA")]
    OAA = data[(data["synthetic_k"] == i) & (data["AA_type"] == "OAA")]
    RBOAA = data[(data["synthetic_k"] == i) & (data["AA_type"] == "RBOAA")]
    
    CAA_NMI_max = []
    TSAA_NMI_max = []
    OAA_NMI_max = []
    RBOAA_NMI_max = []

    CAA_MCC_max = []
    TSAA_MCC_max = []
    OAA_MCC_max = []
    RBOAA_MCC_max = []

    OAA_BDM_min = []
    RBOAA_BDM_min = []

    CAA_loss_min = []
    TSAA_loss_min = []
    OAA_loss_min = []
    RBOAA_loss_min = []

    CAA_k_single = []
    TSAA_k_single = []
    OAA_k_single = []
    RBOAA_k_single = []

    for k in CAA["analysis_k"].unique():
        CAA_NMI_max.append(np.max(CAA[data["analysis_k"] == k]["NMI"]))
        TSAA_NMI_max.append(np.max(TSAA[data["analysis_k"] == k]["NMI"]))
        OAA_NMI_max.append(np.max(OAA[data["analysis_k"] == k]["NMI"]))
        RBOAA_NMI_max.append(np.max(RBOAA[data["analysis_k"] == k]["NMI"]))

        CAA_MCC_max.append(np.max(CAA[data["analysis_k"] == k]["MCC"]))
        TSAA_MCC_max.append(np.max(TSAA[data["analysis_k"] == k]["MCC"]))
        OAA_MCC_max.append(np.max(OAA[data["analysis_k"] == k]["MCC"]))
        RBOAA_MCC_max.append(np.max(RBOAA[data["analysis_k"] == k]["MCC"]))

        CAA_loss_min.append(np.max(CAA[data["analysis_k"] == k]["loss"]))
        TSAA_loss_min.append(np.max(TSAA[data["analysis_k"] == k]["loss"]))
        OAA_loss_min.append(np.max(OAA[data["analysis_k"] == k]["loss"]))
        RBOAA_loss_min.append(np.max(RBOAA[data["analysis_k"] == k]["loss"]))

        OAA_BDM_min.append(np.max(OAA[data["analysis_k"] == k]["BDM"]))
        RBOAA_BDM_min.append(np.max(RBOAA[data["analysis_k"] == k]["BDM"]))
        
        CAA_k_single.append(k)
        TSAA_k_single.append(k)
        OAA_k_single.append(k)
        RBOAA_k_single.append(k)

    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.scatter(RBOAA_k_single,RBOAA_NMI_max, label = "RBOAA")
    plt.plot(RBOAA_k_single,RBOAA_NMI_max,'--')
    plt.scatter(OAA_k_single,OAA_NMI_max, label = "OAA")
    plt.plot(OAA_k_single,OAA_NMI_max,'--')
    # plt.scatter(TSAA_k_single,TSAA_NMI_max, label = "TSAA")
    # plt.plot(TSAA_k_single,TSAA_NMI_max,'--')
    plt.scatter(CAA_k_single,CAA_NMI_max, label = "CAA")
    plt.plot(CAA_k_single,CAA_NMI_max, '--')
    plt.xlabel('Analysis Archetypes', fontsize=14)
    plt.ylabel('NMI', fontsize=14)
    plt.title("NMI over Varying Archetypes w. Synthetic k = {0}".format(i), fontsize = 20)
    plt.legend()
    file = ("NMI_synthetic_k_{0}_varying_archs.png".format(i))
    fig.savefig(os.path.join(path,file),dpi=200)
    plt.close()

    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.scatter(RBOAA_k_single,RBOAA_MCC_max, label = "RBOAA")
    plt.plot(RBOAA_k_single,RBOAA_MCC_max, '--')
    plt.scatter(OAA_k_single,OAA_MCC_max, label = "OAA")
    plt.plot(OAA_k_single,OAA_MCC_max, '--')
    # plt.scatter(TSAA_k_single,TSAA_MCC_max, label = "TSAA")
    # plt.plot(TSAA_k_single,TSAA_MCC_max, '--')
    plt.scatter(CAA_k_single,CAA_MCC_max, label = "CAA")
    plt.plot(CAA_k_single,CAA_MCC_max, '--')
    plt.xlabel('Analysis Archetypes', fontsize=14)
    plt.ylabel('MCC', fontsize=14)
    plt.title("MCC over Varying Archetypes w. Synthetic k = {0}".format(i), fontsize = 20)
    plt.legend()
    file = ("MCC_synthetic_k_{0}_varying_archs.png".format(i))
    fig.savefig(os.path.join(path,file),dpi=200)
    plt.close()


    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.scatter(RBOAA_k_single,RBOAA_BDM_min, label = "RBOAA")
    plt.plot(RBOAA_k_single,RBOAA_BDM_min, '--')
    plt.scatter(OAA_k_single,OAA_BDM_min, label = "OAA")
    plt.plot(OAA_k_single,OAA_BDM_min, '--')
    plt.xlabel('Analysis Archetypes', fontsize=14)
    plt.ylabel('BDM', fontsize=14)
    plt.title("BDM over Varying Archetypes w. Synthetic k = {0}".format(i), fontsize = 20)
    plt.legend()
    file = ("BDM_synthetic_k_{0}_varying_archs.png".format(i))
    fig.savefig(os.path.join(path,file),dpi=200)
    plt.close()


    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.scatter(RBOAA_k_single,RBOAA_loss_min, label = "RBOAA")
    plt.plot(RBOAA_k_single,RBOAA_loss_min, '--')
    plt.scatter(OAA_k_single,OAA_loss_min, label = "OAA")
    plt.plot(OAA_k_single,OAA_loss_min, '--')
    # plt.scatter(TSAA_k_single,TSAA_loss_min, label = "TSAA")
    # plt.plot(TSAA_k_single,TSAA_loss_min, '--')
    plt.scatter(CAA_k_single,CAA_loss_min, label = "CAA")
    plt.plot(CAA_k_single,CAA_loss_min, '--')
    plt.xlabel('Analysis Archetypes', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title("Loss over Varying Archetypes w. Synthetic k = {0}".format(i), fontsize = 20)
    plt.legend()
    file = ("LOSS_synthetic_k_{0}_varying_archs.png".format(i))
    fig.savefig(os.path.join(path,file),dpi=200)
    plt.close()
