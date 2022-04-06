########## IMPORT ##########
from turtle import color
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


########## PLOTS CLASS ##########
class _plots:

    def _PCA_scatter_plot(self,Z,X):
        
        pca = PCA(n_components=2)
        pca.fit(Z.T)
        
        Z_pca = pca.transform(Z.T)
        X_pca = pca.transform(X.T)
        
        plt.rcParams["figure.figsize"] = (10,10)
        plt.scatter(X_pca[:,0], X_pca[:,1], c ="black", s = 1)
        plt.scatter(Z_pca[:,0], Z_pca[:,1], marker ="^", c ="#2c6c8c", s = 750, label="Archetypes")
        plt.xlabel("Principal Component 1", fontsize=25)
        plt.ylabel("Principal Component 2", fontsize=25)
        plt.legend(prop={'size': 15})
        
        #plt.savefig("Scatter-plot Principal Component")
        plt.show()


    def _attribute_scatter_plot(self,Z,X,attributes):
        
        plt.rcParams["figure.figsize"] = (10,10)
        plt.scatter(X[attributes[0],:], X[attributes[1],:], c ="black", s = 1)
        plt.scatter(Z[attributes[0],:], Z[attributes[1],:], marker ="^", c ="#2c6c8c", s = 750, label="Archetypes")
        plt.xlabel(f"Attribute {attributes[0]}", fontsize=25)
        plt.ylabel(f"Attribute {attributes[1]}", fontsize=25)
        plt.legend(prop={'size': 15})
        
        #plt.savefig("Scatter-plot Principal Component")
        plt.show()


    def _loss_plot(self,loss,type):
        
        plt.plot(loss, c="#2c6c8c")
        plt.xlabel(f"Itteration of {type}")
        plt.ylabel(f"Loss of {type}")
        plt.title(f"Loss w.r.t. Itteration of {type}")
        plt.show()


    def _mixture_plot(self,Z,A,type):

        plt.rcParams["figure.figsize"] = (10,10)
        fig = plt.figure()
        ax = fig.add_subplot(111)

        K = len(Z.T)
        corners = []
        for k in range(K):
            corners.append([np.cos(((2*np.pi)/K)*(k)), np.sin(((2*np.pi)/K)*(k))])
            plt.plot(np.cos(((2*np.pi)/K)*(k)), np.sin(((2*np.pi)/K)*(k)), marker="o", markersize=10, markeredgecolor="black", markerfacecolor="#2c6c8c")

        points_x = []
        points_y = []
        for p in A.T:
            x = 0
            y = 0
            for k in range(K):
                x += p[k] * np.cos(((2*np.pi)/K)*(k))
                y += p[k] * np.sin(((2*np.pi)/K)*(k))
            points_x.append(x)
            points_y.append(y)
        
        p = Polygon(corners, closed=False)
        #ax = plt.gca()
        ax.add_patch(p)
        ax.set_xlim(-1.1,1.1)
        ax.set_ylim(-1.1,1.1)
        ax.set_aspect('equal')
        plt.scatter(points_x, points_y, c ="black", s = 1)
        plt.title(f"Mixture Plot of {type}")
        plt.show()


    def _barplot(self,Z,columns,archetype_number,type):
        plt.rcParams["figure.figsize"] = (10,10)
        archetype = Z.T[archetype_number]
        fig, ax = plt.subplots()
        ax.set_ylabel('Value')
        ax.set_title(f"Archeype {archetype_number}")
        ax.bar(np.arange(len(archetype)),archetype)
        ax.set_xticks(np.arange(len(archetype)))
        ax.set_xticklabels(labels=columns)
        if type == "CAA":
            plt.ylim(1, np.max(Z+0.2))
        else:
            plt.ylim(0, np.max(Z+0.2))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig.set_size_inches(10, 10)
        plt.show()


    def _barplot_all(self,Z,columns):
        plt.rcParams["figure.figsize"] = (10,10)
        data = []
        names = ["Attributes"]
        
        for (arch, column) in zip(Z,columns):
            current_data = [column]
            for value in arch:
                current_data.append(value)
                
            data.append(current_data)
        
        for i in range(len(Z.T)):
            names.append("Archetype {0}".format(i+1))

        df=pd.DataFrame(data,columns=names)
        df.plot(x="Attributes", y=names[1:], kind="bar",figsize=(10,10))
        plt.show()


    def _typal_plot(self, Z, types, weighted):
        plt.rcParams["figure.figsize"] = (10,10)
        fig, ax = plt.subplots()
        type_names = types.keys()
        type_names_display = list(types.keys())
        labels = [f"Archetype {i}" for i in range(len(Z.T))]
        width = 0.5
        bottoms = []
        bottoms.append([0 for i in range(len(Z.T))])
        values = []
        for label in type_names:
            label_values = []
            for archetype in Z.T:
                archetype_value = 0
                for i in types[label]:
                    archetype_value += archetype[i]
                if weighted in ["equal","equal_norm"]:
                    archetype_value = archetype_value / len(types[label])
                label_values.append(archetype_value)
            values.append(label_values)
        
        values_new = np.array(values)

        if weighted in ["norm","equal_norm"]:
            for i in range(len(values)):
                values_new[i] = values_new[i] / np.sum(values,0)

        for i in range(len(values)-1):
            bottoms.append([b + l for (b,l) in zip(bottoms[-1],values_new[i])])

        for i in range(len(values)):
            ax.bar(labels, values_new[i], width, bottom=bottoms[i], label=type_names_display[i])
        ax.set_ylabel('Value')
        ax.set_title('Typal Composition of Archetypes')
        ax.legend()

        plt.show()
        
