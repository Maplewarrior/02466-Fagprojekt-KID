import os
import pickle


results = {}
directory = 'synthetic_results'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        file = open(filepath,'rb')
        result = pickle.load(file)

        AA_type = filename.split("_")[0]
        sigma = float(filename.split("_")[2])
        k = int(filename.split("_")[4])
        a = float(filename.split("_")[6])
        b = float(filename.split("_")[8])
        
        if not AA_type in results:
            results[AA_type] = {}
        if not sigma in results[AA_type]:
            results[AA_type][sigma] = {}
        if not k in results[AA_type][sigma]:
            results[AA_type][sigma][k] = {}
        if not a in results[AA_type][sigma][k]:
            results[AA_type][sigma][k][a] = {}
        if not b in results[AA_type][sigma][k][a]:
            results[AA_type][sigma][k][a][b] = {}
        if "metadata" in filename:
            if not "metadata" in results[AA_type][sigma][k][a][b]:
                results[AA_type][sigma][k][a][b]["metadata"] = []
            results[AA_type][sigma][k][a][b]["metadata"].append(result)
        elif not "metadata" in filename:
            if not "analysis" in results[AA_type][sigma][k][a][b]:
                results[AA_type][sigma][k][a][b]["analysis"] = []
            results[AA_type][sigma][k][a][b]["analysis"].append(result)
        
        print(AA_type + "_" + str(sigma) + "_" + str(k) + "_" + str(a) + "_" + str(b))

print(results["CAA"][-0.43][3][0.85][1.0])