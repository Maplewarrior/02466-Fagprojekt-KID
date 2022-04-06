########## IMPORT ##########
from matplotlib.pyplot import plot
from plots_class import _plots
import pickle


########## CONVENTIONAL ARCHETYPAL ANALYSIS RESULT ##########
class _CAA_result:

    plots = _plots()
    
    def __init__(self, A, B, X, X_hat, n_iter, RSS, Z, K, time, columns,type):
        self.A = A
        self.B = B
        self.X = X
        self.X_hat  = X_hat
        self.n_iter = n_iter
        self.RSS = RSS
        self.Z = Z
        self.K = K
        self.time = time
        self.columns = columns
        self.type = type

    def _print(self):
        print("/////////////// INFORMATION ABOUT CONVENTIONAL ARCHETYPAL ANALYSIS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
        print(f"▣ The Conventional Archetypal Analysis was computed using {self.K} archetypes.")
        print(f"▣ The Conventional Archetypal Analysis was computed on {len(self.X)} attributes.")
        print(f"▣ The Conventional Archetypal Analysis was computed on {len(self.X[0,:])} subjects.")
        print(f"▣ The Conventional Archetypal Analysis ran for {self.n_iter} itterations.")
        print(f"▣ The Conventional Archetypal Analysis took {self.time} seconds to complete.")
        print(f"▣ The final RSS was: {self.RSS[-1]}.")

    def _plot(self,plot_type, attributes, archetype_number, types, weighted):
        
        if plot_type == "PCA_scatter_plot":
            self.plots._PCA_scatter_plot(self.Z,self.X_hat)
        elif plot_type == "attribute_scatter_plot":
            self.plots._attribute_scatter_plot(self.Z,self.X_hat,attributes)
        elif plot_type == "loss_plot":
            self.plots._loss_plot(self.RSS,self.type)
        elif plot_type == "mixture_plot":
            self.plots._mixture_plot(self.Z,self.A,self.type)
        elif plot_type == "barplot":
            self.plots._barplot(self.Z,self.columns,archetype_number,self.type)
        elif plot_type == "barplot_all":
            self.plots._barplot_all(self.Z,self.columns)
        elif plot_type == "typal_plot":
            self.plots._typal_plot(self.Z,types,weighted)

    def _save(self,filename):
        file = open("results/" + self.type + "_" + filename + '.obj','wb')
        pickle.dump(self, file)
        file.close()


########## ORDINAL ARCHETYPAL ANALYSIS RESULT ##########
class _OAA_result:

    plots = _plots()
    
    def __init__(self, A, B, X, n_iter, b, Z, X_tilde, Z_tilde, X_hat, loss, K, time, columns,type):
        self.A = A
        self.B = B
        self.X = X
        self.n_iter = n_iter
        self.b = b
        self.X_tilde = X_tilde
        self.Z_tilde = Z_tilde
        self.X_hat = X_hat
        self.loss = loss
        self.Z = Z
        self.K = K
        self.time = time
        self.columns = columns
        self.type = type

    def _print(self):
        print("/////////////// INFORMATION ABOUT ORDINAL ARCHETYPAL ANALYSIS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
        print(f"▣ The Ordinal Archetypal Analysis was computed using {self.K} archetypes.")
        print(f"▣ The Ordinal Archetypal Analysis was computed on {len(self.X)} attributes.")
        print(f"▣ The Ordinal Archetypal Analysis was computed on {len(self.X[0,:])} subjects.")
        print(f"▣ The Ordinal Archetypal Analysis ran for {self.n_iter} itterations.")
        print(f"▣ The Ordinal Archetypal Analysis took {self.time} seconds to complete.")
        print(f"▣ The final loss was: {self.loss[-1]}.")
    
    def _plot(self,plot_type, attributes, archetype_number, types, weighted):
        
        if plot_type == "PCA_scatter_plot":
            self.plots._PCA_scatter_plot(self.Z_tilde,self.X_hat)
        elif plot_type == "attribute_scatter_plot":
            self.plots._attribute_scatter_plot(self.Z_tilde,self.X_hat,attributes)
        elif plot_type == "loss_plot":
            self.plots._loss_plot(self.loss,self.type)
        elif plot_type == "mixture_plot":
            self.plots._mixture_plot(self.Z,self.A,self.type)
        elif plot_type == "barplot":
            self.plots._barplot(self.Z_tilde,self.columns,archetype_number,self.type)
        elif plot_type == "barplot_all":
            self.plots._barplot_all(self.Z_tilde,self.columns)
        elif plot_type == "typal_plot":
            self.plots._typal_plot(self.Z_tilde,types,weighted)

    def _save(self,filename):
        file = open("results/" + self.type + "_" + filename + '.obj','wb')
        pickle.dump(self, file)
        file.close()