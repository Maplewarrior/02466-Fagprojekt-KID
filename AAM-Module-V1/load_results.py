import os
import pickle
from collections import defaultdict


results = {}

directory = 'synthetic_results'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        file = open(filepath,'rb')
        result = pickle.load(file)

        AA_type = filename.split("_")[0]
        sigma = float(filename.split("_")[3])
        k = int(filename.split("_")[5].replace(".obj",""))
        
        if not AA_type in results:
            results[AA_type] = {}
        if not sigma in results[AA_type]:
            results[AA_type][sigma] = {}
        if not k in results[AA_type][sigma]:
            results[AA_type][sigma][k] = {}
        if "metadata" in filename:
            if not "metadata" in results[AA_type][sigma][k]:
                results[AA_type][sigma][k]["metadata"] = []
            results[AA_type][sigma][k]["metadata"].append(result)
        elif not "metadata" in filename:
            if not "analysis" in results[AA_type][sigma][k]:
                results[AA_type][sigma][k]["analysis"] = []
            results[AA_type][sigma][k]["analysis"].append(result)

