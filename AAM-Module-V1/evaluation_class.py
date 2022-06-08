########## IMPORT ##########
import numpy as np
import os
import pickle

########## EVALUATION CLASS ##########
class _evaluation:

    def load_results(self):
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

        return results

    def _matrix_correlation_coefficient(self, A1, A2):
        K, _ = A1.shape() #kolonner - btw A1 og A2 skal have samme antal kolonner aka archetyper
        corr = np.zeros((K,K))
        for i in range(K):
            for j in range(K):
                corr[i][j] = np.corrcoef(A1[:,i], A2[:,j])[0][1]
        
        max_list = []
        for _ in range(K):
            row, column = np.unravel_index(corr.argmax(), corr.shape)
            max_list.append(corr[row][column])
            corr = np.delete(corr, row, axis=0)
            corr = np.delete(corr, column, axis=1)
        
        return np.mean(max_list)

    def _calcMI(self, A1, A2):
        P = A1@A2.T
        PXY = P/sum(sum(P))
        PXPY = np.outer(np.sum(PXY,axis=1), np.sum(PXY,axis=0))
        ind = np.where(PXY>0)
        MI = sum(PXY[ind]*np.log(PXY[ind]/PXPY[ind]))
        return MI

    def _normalised_mutual_information(self,A1,A2):
        #krav at værdierne i række summer til 1 ???
        NMI = (2*self._calcMI(A1,A2)) / (self._calcMI(A1,A1) + self._calcMI(A2,A2))
        return NMI

    def _resbonse_bias_analysis(self,b1,b2):
        total = 0
        k = b1.shape[0]
        for i in range(k):
            total += abs(b1[i]-b2[i])
        return total/k

    def test(self):
        results = self.load_results()


        a_an = results["TSAA"][-20][5][1][1000]["analysis"][-1].A
        a_gt = results["TSAA"][-20][5][1][1000]["metadata"][-1].A

        print(self._normalised_mutual_information(a_an,a_gt))

#ev = _evaluation()
#ev.test()