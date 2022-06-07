########## IMPORT ##########
import numpy as np

########## EVALUATION CLASS ##########
class _evaluation:

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
