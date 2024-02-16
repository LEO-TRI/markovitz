import numpy as np
from scipy import stats


class CorrelationDenoiserPca:
    """
    Use init_from_z_scores to instantiate class from z_scores
    """
    
    def __init__(self, 
                 C: np.ndarray, 
                 variable_names: list) -> None:
        
        self.C = C
        self.variable_names = list(variable_names)
        
        #Filled by specific methods
        self.explained_variance = None
        self.denoised_C = None
    
    @classmethod
    def init_from_z_scores(cls, dict_z_score : dict, bias: bool=False) -> "CorrelationDenoiserPca":
        """
        Direct computation of the correlation matrix with the product of Z-score matrices.

        Only necessary if we receive Z scores -> Class be instantiated directly with C as well

        Parameters
        ----------
        dict_z_score : dict
             A dict of format factor : z_scores
        bias : bool
            Whether to calculate covariance matrix with bias, by default False, 
            means we divide by N - 1 and not N

        Returns
        -------
        CorrelationDenoiserPca
            An instantiated denoiser
        """
        
        list_market_variable = list(dict_z_score.keys())
        Z = np.asarray(list(dict_z_score.values())) #Dims (N_features, T)
        #z_list = [np.array(dict_z_score[variable]) for variable in list_market_variable] 
            
        V = (Z @ Z.T) / (Z.shape[1] - int(not bias)) #N -1 is calculated to match np.corrcoef implementation if bias is False
        S = np.diag(V) ** 0.5 #Isolating the sigmas from the variances
        C = V / S[None,:] / S[:,None] #Broadcasting trick so that V is divided by matrix (1,len(V)) and (len(V),1) -> taken from np.corrcoef formula
        
        #C = np.corrcoef(Z, rowvar=False)
        
        return CorrelationDenoiserPca(C=C, variable_names=list_market_variable)
    
    @staticmethod
    def _select_n_eigen_components(C: np.ndarray = None, 
                                   target_variance: float=0.8) -> int:
        """
        Inside function, should not be used outside the instance methods
        
        Calculate the number of eigenvalues necessary to reach target variance
        
        Parameters
        ----------
        C : np.ndarray, optional
            Correlation matrix, by default None
        target_variance : float, optional
            The target variance to reach, by default 0.8

        Returns
        -------
        int
            The number n of eigenvalues to keep
            
        Notes
        -------
        Will always select n eigenvalues so that explained_variance >= target_variance
        """
                        
        eigenvalues, _ = np.linalg.eig(C)
        
        cumulated_variance = np.cumsum(eigenvalues) #Each value represents the explained variance by adding 1 more eigenvalue
        overall_variance = cumulated_variance[-1] #Last val of the cumsum is the overall sum
        #We take the fraction of the cumsum / sum and remove the target_variance -> The indice of the value closest to 0 and > 0 is
        #the best n to have the most variance explained while denoising. 
        cumulated_explained_variance = (cumulated_variance / overall_variance) - target_variance
        #Setting a high number to avoid selecting a negative number close to 0 -> All values are now >= 0
        cumulated_explained_variance[cumulated_explained_variance < 0] = 99
        
        n_eigen_components = np.argmin(cumulated_explained_variance)
        
        return n_eigen_components
    
    def _shared_eigen_calculation(self,
                                 C : np.ndarray,
                                 n_eigen_components: int=None,
                                 target_variance: float=None,) -> tuple[np.ndarray, np.ndarray, int]:
        
        if (n_eigen_components is None) & (target_variance is None):
            raise ValueError("You must pass either a number of eigenvalues to keep or a target variance")
        elif (n_eigen_components is not None) & (target_variance is not None):
            raise ValueError("You can only pass one of n_eigen_components or target_variance")
            
        if (n_eigen_components is None): #If true, implies target_variance is passed as argument thanks to the above checks
            n_eigen_components = self._select_n_eigen_components(C, target_variance=target_variance)
            
        eigenvalues, P = np.linalg.eig(C)
        eigenvalues = np.real(eigenvalues)
        P = np.real(P)
        
        return (eigenvalues, P, n_eigen_components)

    def get_eigen_elements(self, 
                           C: np.ndarray=None,
                           n_eigen_components: int=None,
                           target_variance: float=None,) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple of the eigenvalues and their corresponding eigenvectors


        Parameters
        ----------
        C : np.ndarray, optional
            Correlation matrix, by default None. If none, will attempt to use C used when
            instantiating
        n_eigen_components : int, optional
            Number of eigen-values and vectors to keep, by default None
        target_variance : float, optional
            The target variance to reach with the number of eigen- values and vectors to keep, by default None

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple with format eigenvalues, eigenvectors
            
        Notes
        -------
        You can pass either n_eigen_components or target_variance, but not both at the same time
        """        

        if C is None:
            C = self.C.copy()

        eigenvalues, P, n_eigen_components = self._shared_eigen_calculation(C, n_eigen_components, target_variance)
        
        #NOTE: Check the ordering of eigenvalues -> Implemented with a rankdata and a boolean mask
        #We create a mask based on the ranking of eigenvalues. All eigenvalues with ranks above len(eigenvalues) - n_eigen_components 
        #are kept
        #Ordinal method means that if there are a tie (and there will be since eigenvalues will be often 1), the tie is solved by 
        #first come first served
        #stats.rankdata is useful bc it handles ties better than pure np. 

        mask_indices = stats.rankdata(eigenvalues, method='ordinal') > (len(eigenvalues) - n_eigen_components) 
        eigenvalues = eigenvalues[mask_indices] #Slicing out eigenvalues whose ranks are above n_eigen_components
        P = P[:,mask_indices] #Slicing the equivalent eigenvectors from 
        
        return (eigenvalues, P)

    def compute_denoised_C(self, 
                            C: np.ndarray=None, 
                            n_eigen_components: int=None,
                            target_variance: float=None,
                            inplace: bool=True) -> "CorrelationDenoiserPca":
        """
        Denoise a C matrix based on a PCA approach

        Parameters
        ----------
        C : np.ndarray
            Correlation matrix, must be square
        n_eigen_components : int, optional
            Number of eigenvalues to keep, by default None
        target_variance : float, optional
            Target variance to keep, by default None
        inplace : bool, optional
            Whether to return the instance or the new matrix, by default True

        Returns
        -------
        CorrelationDenoiserPca
            Updated denoiser carrying a denoised C matrix
            
        Notes
        -------
        n_eigen_components & target_variance cannot be passed as arguments at the same time
        
        You either pass a specific n_eigen_components value, or it will be selected based on 
        target_variance
        """
                        
        if C is None:
            C = self.C.copy()

        eigenvalues, P, n_eigen_components = self._shared_eigen_calculation(C, n_eigen_components, target_variance)

        sum_values = np.sum(eigenvalues) #Sum of all eigenvalues

        #Creating a mask to force 0 on the eigenvalues below the target
        mask_indices = stats.rankdata(eigenvalues, method='ordinal') <= n_eigen_components
        eigenvalues[mask_indices] = 0 #Setting all unneeded eigenvalues to 0

        sum_eigen = np.sum(eigenvalues) #Sum of eigenvalues kept to reach the target variance
        
        D = np.diag(eigenvalues) #Creates a new matrix with the diag filled in and the rest of values being 0
        inv_P = np.linalg.inv(P)  
        C_bis =  np.real(P @ D @ inv_P) #Since part of the diag is 0, parts of the data will be lost
        
        diag_indices = np.arange(C_bis.shape[1])
        C_bis[diag_indices, diag_indices] = 1 #Re-setting the diagonal values to 1
        
        if inplace: 
            self.explained_variance = (sum_eigen / sum_values)
            self.denoised_C = C_bis
            return self
        
        return C_bis
    
        
        