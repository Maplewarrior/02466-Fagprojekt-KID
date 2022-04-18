########## IMPORTS ##########
from CAA_class import _CAA
from OAA_class import _OAA
from RBOAA_class import _RBOAA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from os import path


########## ARCHETYPAL ANALYSIS MODULE CLASS ##########
class AA:
    
    def __init__(self):
        
        self._CAA = _CAA()
        self._OAA = _OAA()
        self._RBOAA = _RBOAA()
        self._results = {"CAA": [], "OAA": [], "RBOAA": []}
        self._has_data = False


    def load_data(self, X: np.ndarray, columns: list[str]):
        self.columns = columns
        self.X = X
        self.N, self.M = X.shape
        self._has_data = True
        if self.N>self.M:
            print("Your data has more attributes than subjects.")
            print(f"Your data has {self.M} attributes and {self.N} subjects.")
            print("This is highly unusual for this type of data.")
            print("Please try loading transposed data instead.")
        else:
            print(f"\nThe data was loaded successfully!\n")


    def load_csv(self, filename: str, columns: list[int] = None, rows: int = None):
        self.columns, self.M, self.N, self.X = self._clean_data(filename, columns, rows)
        self._has_data = True
        print(f"\nThe data of \'{filename}\' was loaded successfully!\n")

    
    def _clean_data(self, filename, columns, rows):
        df = pd.read_csv(filename)

        column_names = df.columns.to_numpy()
        
        if not columns is None:
            column_names = column_names[columns]
            X = df[column_names]
        else:
            X = df[column_names]
        if not rows is None:
            X = X.iloc[range(rows),:]

        X = X.to_numpy().T
        M, N = X.shape

        return column_names, M, N, X
    

    def analyse(self, K: int = 3, n_iter: int = 1000, AA_type = "all", lr: float = 0.001, mute: bool = False):
        if self._has_data:
            if AA_type == "all" or AA_type == "CAA":
                self._results["CAA"].insert(0,self._CAA._compute_archetypes(self.X, K, n_iter, lr, mute,self.columns))
            elif AA_type == "all" or AA_type == "OAA":
                self._results["OAA"].insert(0,self._OAA._compute_archetypes(self.X, K, n_iter, lr, mute,self.columns))
            elif AA_type == "all" or AA_type == "RBOAA":
                self._results["RBOAA"].insert(0,self._RBOAA._compute_archetypes(self.X, K, n_iter, lr, mute,self.columns))
            else:
                print("The AA_type \"{0}\" specified, does not match any of the possible AA_types.".format(AA_type))
        
        else:
            print("\nYou have not loaded any data yet! \nPlease load data through the \'load_data\' or \'load_csv\' methods and try again.\n")


    def plot(self, 
            model_type: str = "CAA", 
            plot_type: str = "PCA_scatter_plot", 
            result_number: int = 0, 
            attributes: list[int] = [0,1], 
            archetype_number: int = 0, 
            types: dict = {},
            weighted: str = "equal_norm"):
        
        if not model_type in ["CAA", "OAA", "RBOAA"]:
            print("\nThe model type you have specified can not be recognized. Please try again.")
        elif not plot_type in ["PCA_scatter_plot","attribute_scatter_plot","loss_plot","mixture_plot","barplot","barplot_all","typal_plot"]:
            print("\nThe plot type you have specified can not be recognized. Please try again.\n")
        elif result_number < 0 or not result_number < len(self._results[model_type]):
            print("\nThe result you are requesting to plot is not availabe.\n Please make sure you have specified the input correctly.\n")
        elif not weighted in ["none","equal_norm","equal","norm"]:
            print(f"\nThe \'weighted\' parameter recieved an unexpected value of {weighted}.\n")
        elif any(np.array(attributes) < 0) or any(np.array(attributes) > len(self._results[model_type][result_number].columns)):
            print(f"\nThe \'attributes\' parameter recieved an unexpected value of {attributes}.\n")
        elif archetype_number < 0 or archetype_number > self._results[model_type][result_number].K:
            print(f"\nThe \'archetype_number\' parameter recieved an unexpected value of {archetype_number}.\n")
        
        else:
            result = self._results[model_type][result_number]
            result._plot(plot_type,attributes,archetype_number,types,weighted)
            print("\nThe requested plot was successfully plotted!\n")


    def save_analysis(self,filename: str = "analysis",model_type: str = "CAA", result_number: int = 0):

        if not model_type in ["CAA", "OAA", "RBOAA"]:
            print("\nThe model type you have specified can not be recognized. Please try again.\n")
        elif not result_number < len(self._results[model_type]):
            print("\nThe analysis you are requesting to save is not availabe.\n Please make sure you have specified the input correctly.\n")
        
        
        else:
            self._results[model_type][result_number]._save(filename)
            print("\nThe analysis was successfully saved!\n")

    
    def load_analysis(self, filename: str = "analysis", model_type: str = "CAA"):
        if not model_type in ["CAA", "OAA", "RBOAA"]:
            print("\nThe model type you have specified can not be recognized. Please try again.\n")
        elif not path.exists("results/" + model_type + "_" + filename + '.obj'):
            print(f"The analysis {filename} of type {model_type} does not exist on your device.")
        
        
        else:
            file = open("results/" + model_type + "_" + filename + '.obj','rb')
            result = pickle.load(file)
            file.close()
            self._results[model_type].append(result)
            print("\nThe analysis was successfully loaded!\n")