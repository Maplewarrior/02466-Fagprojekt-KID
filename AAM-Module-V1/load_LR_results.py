import pandas as pd
import os
import pickle


results = {"TYPE": [], "LR": [], "DataSize": [], "Loss": []}
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
            results["TYPE"].append(AA_type)
            results["LR"].append(LR)
            results["DataSize"].append(data_size)
            results["Loss"].append(loss)

print(results["Loss"])


        
