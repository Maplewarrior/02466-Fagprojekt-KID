########## IMPORTS ##########
from CAA_class import _CAA
from OAA_class import _OAA
from RBOAA_class import _RBOAA
from TSAA_class import _TSAA
from synthetic_data_class import _synthetic_data

#### TJEK v
# from evaluation_class import _evaluation

import pandas as pd
import numpy as np
import pickle
from os import path


########## ARCHETYPAL ANALYSIS MODULE CLASS ##########
class AA:
    
    def __init__(self):
        
        self._CAA = _CAA()
        self._OAA = _OAA()
        self._RBOAA = _RBOAA()
        self._TSAA = _TSAA()
        self._results = {"CAA": [], "OAA": [], "RBOAA": [], "TSAA": []}
        self._synthetic_results = {"CAA": [], "OAA": [], "RBOAA": [], "TSAA": []}
        self._has_data = False
        self.has_synthetic_data = False


    def load_data(self, X: np.ndarray, columns: list()):
        self.columns = columns
        self.X = X
        self.N, self.M = X.shape
        self._has_data = True
        if self.N<self.M:
            print("Your data has more attributes than subjects.")
            print(f"Your data has {self.M} attributes and {self.N} subjects.")
            print("This is highly unusual for this type of data.")
            print("Please try loading transposed data instead.")
        else:
            print(f"\nThe data was loaded successfully!\n")


    def load_csv(self, filename: str, columns: list(), rows: int = None):
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
    

    def create_synthetic_data(self, N: int = 1000, M: int = 10, K: int = 3, p: int = 6, sigma: int = 1, rb: bool = False, b_param: int = 100, a_param: int = 1, mute = False):
        if N < 2:
            print("The value of N can't be less than 2. The value specified was {0}".format(N))
        elif M < 2:
            print("The value of M can't be less than 2. The value specified was {0}".format(M))
        elif K < 2:
            print("The value of K can't be less than 2. The value specified was {0}".format(K))
        elif p < 2:
            print("The value of p can't be less than 2. The value specified was {0}".format(p))
        else:
            self._synthetic_data = _synthetic_data(N, M, K, p, sigma, rb, a_param, b_param)
            self.has_synthetic_data = True
            self._synthetic_results = {"CAA": [], "OAA": [], "RBOAA": [], "TSAA": []}
            if not mute:
                print("\nThe synthetic data was successfully created! To use the data in an analysis, specificy the with_synthetic_data parameter as True.\n")


    def analyse(self, K: int = 3, p: int = 6, n_iter: int = 1000, early_stopping: bool = False, AA_type = "all", lr: float = 0.1, mute: bool = False, with_synthetic_data: bool = False, with_hot_start: bool = False):
        if self._has_data and not with_synthetic_data:
            if AA_type == "all" or AA_type == "CAA":
                self._results["CAA"].insert(0,self._CAA._compute_archetypes(self.X, K, n_iter, lr, mute,self.columns,early_stopping=early_stopping))
            elif AA_type == "all" or AA_type == "OAA":
                self._results["OAA"].insert(0,self._OAA._compute_archetypes(self.X, K, p, n_iter, lr, mute,self.columns,early_stopping=early_stopping,with_CAA_initialization=with_hot_start))
            elif AA_type == "all" or AA_type == "RBOAA":
                self._results["RBOAA"].insert(0,self._RBOAA._compute_archetypes(self.X, K, p, n_iter, lr, mute,self.columns,early_stopping=early_stopping, with_OAA_initialization = with_hot_start))
            elif AA_type == "all" or AA_type == "TSAA":
                self._results["TSAA"].insert(0,self._TSAA._compute_archetypes(self.X, K, n_iter, lr, mute,self.columns,early_stopping=early_stopping))
            else:
                print("The AA_type \"{0}\" specified, does not match any of the possible AA_types.".format(AA_type))
    
        elif self.has_synthetic_data and with_synthetic_data:
            if AA_type == "all" or AA_type == "CAA":
                self._synthetic_results ["CAA"].insert(0,self._CAA._compute_archetypes(self._synthetic_data.X, K, n_iter, lr, mute, self._synthetic_data.columns, with_synthetic_data=True,early_stopping=early_stopping))
            elif AA_type == "all" or AA_type == "OAA":
                self._synthetic_results["OAA"].insert(0,self._OAA._compute_archetypes(self._synthetic_data.X, K, p, n_iter, lr, mute, self._synthetic_data.columns,with_synthetic_data=True,early_stopping=early_stopping,with_CAA_initialization = with_hot_start))
            elif AA_type == "all" or AA_type == "RBOAA":
                self._synthetic_results["RBOAA"].insert(0,self._RBOAA._compute_archetypes(self._synthetic_data.X, K, p, n_iter, lr, mute, self._synthetic_data.columns,with_synthetic_data=True,early_stopping=early_stopping,with_OAA_initialization = with_hot_start))
            elif AA_type == "all" or AA_type == "TSAA":
                self._synthetic_results["TSAA"].insert(0,self._TSAA._compute_archetypes(self._synthetic_data.X, K, n_iter, lr, mute, self._synthetic_data.columns,with_synthetic_data=True,early_stopping=early_stopping))
            else:
                print("The AA_type \"{0}\" specified, does not match any of the possible AA_types.".format(AA_type))
        else:
            print("\nYou have not loaded any data yet! \nPlease load data through the \'load_data\' or \'load_csv\' methods and try again.\n")


    def plot(self, 
            model_type: str = "CAA", 
            plot_type: str = "PCA_scatter_plot", 
            result_number: int = 0, 
            attributes: list() = [0,1], 
            archetype_number: int = 0, 
            types: dict = {},
            weighted: str = "equal_norm",
            with_synthetic_data: bool = False):
        
        if not model_type in ["CAA", "OAA", "RBOAA", "TSAA"]:
            print("\nThe model type you have specified can not be recognized. Please try again.")
        elif not plot_type in ["PCA_scatter_plot","attribute_scatter_plot","loss_plot","mixture_plot","barplot","barplot_all","typal_plot"]:
            print("\nThe plot type you have specified can not be recognized. Please try again.\n")
        elif not weighted in ["none","equal_norm","equal","norm"]:
            print(f"\nThe \'weighted\' parameter received an unexpected value of {weighted}.\n")

        elif not with_synthetic_data:
            if result_number < 0 or not result_number < len(self._results[model_type]):
                print("\nThe result you are requesting to plot is not availabe.\n Please make sure you have specified the input correctly.\n")
            elif archetype_number < 0 or archetype_number > self._results[model_type][result_number].K:
                print(f"\nThe \'archetype_number\' parameter received an unexpected value of {archetype_number}.\n")
            elif any(np.array(attributes) < 0) or any(np.array(attributes) > len(self._results[model_type][result_number].columns)):
                print(f"\nThe \'attributes\' parameter received an unexpected value of {attributes}.\n")
            else:
                result = self._results[model_type][result_number]
                result._plot(plot_type,attributes,archetype_number,types,weighted)
                print("\nThe requested plot was successfully plotted!\n")
        
        else:
            if result_number < 0 or not result_number < len(self._synthetic_results[model_type]):
                print("\nThe result you are requesting to plot is not available.\n Please make sure you have specified the input correctly.\n")
            elif archetype_number < 0 or archetype_number > self._synthetic_results[model_type][result_number].K:
                print(f"\nThe \'archetype_number\' parameter received an unexpected value of {archetype_number}.\n")
            elif any(np.array(attributes) < 0) or any(np.array(attributes) > len(self._synthetic_results[model_type][result_number].columns)):
                print(f"\nThe \'attributes\' parameter received an unexpected value of {attributes}.\n")
            else:
                result = self._synthetic_results[model_type][result_number]
                result._plot(plot_type,attributes,archetype_number,types,weighted)
                print("\nThe requested synthetic data result plot was successfully plotted!\n")


    def save_analysis(self,filename: str = "analysis",model_type: str = "CAA", result_number: int = 0, with_synthetic_data: bool = False, save_synthetic_data: bool = True):

        if not model_type in ["CAA", "OAA", "RBOAA", "TSAA"]:
            print("\nThe model type you have specified can not be recognized. Please try again.\n")
        
        if not with_synthetic_data:
            if not result_number < len(self._results[model_type]):
                print("\nThe analysis you are requesting to save is not available.\n Please make sure you have specified the input correctly.\n")
            
            
            else:
                self._results[model_type][result_number]._save(filename)
                print("\nThe analysis was successfully saved!\n")
        else:
            if not result_number < len(self._synthetic_results[model_type]):
                print("\nThe analysis with synthetic data, which you are requesting to save is not available.\n Please make sure you have specified the input correctly.\n")
            
            else:
                self._synthetic_results[model_type][result_number]._save(filename)
                if save_synthetic_data:
                    self._synthetic_data._save(model_type,filename)
                print("\nThe analysis was successfully saved!\n")

    
    def load_analysis(self, filename: str = "analysis", model_type: str = "CAA", with_synthetic_data: bool = False):
        if not model_type in ["CAA", "OAA", "RBOAA", "TSAA"]:
            print("\nThe model type you have specified can not be recognized. Please try again.\n")

        elif not with_synthetic_data:
            if not path.exists("results/" + model_type + "_" + filename + '.obj'):
                print(f"The analysis {filename} of type {model_type} does not exist on your device.")
            
            
            else:
                file = open("results/" + model_type + "_" + filename + '.obj','rb')
                result = pickle.load(file)
                file.close()
                self._results[model_type].append(result)
                print("\nThe analysis was successfully loaded!\n")

        else:
            if not path.exists("synthetic_results/" + model_type + "_" + filename + '.obj'):
                print(f"The analysis {filename} with synthetic data of type {model_type} does not exist on your device.")
            
            else:
                file = open("synthetic_results/" + model_type + "_" + filename + '.obj','rb')
                result = pickle.load(file)
                file.close()
                self._synthetic_results[model_type].append(result)

                file = open("synthetic_results/" + model_type + "_" + filename + '_metadata' + '.obj','rb')
                result = pickle.load(file)
                file.close()
                self._synthetic_data = result

                print("\nThe analysis with synthetic data was successfully loaded!\n")
                self.has_synthetic_data = True


    """def evaluate(self, ):

        # tjek størrelsen af matricerne passer - sådan at der er lige mange kolonner i A1 og A2
        self._evaluation._matrix_correlation_coefficient(A1,A2)

        # tjek at kolonner summerer til 1
        self._evaluation._normalised_mutual_information(A1,A2)

        # tjek at endepunkter 0 og 1 ikke er med som grænser
        # for RBOAA har vi mange lister af boundaries - dette skal løses på en måde, da vi pt. kun kan sammenligne 2 lister
        self._evaluation._resbonse_bias_analysis(b1,b2)
    """

