import numpy as np


class WhiteNoiseSampler:
    """
    Sampler for white noise
    
    Supports two methods, PCA decomposition or Cholesky decomposition
    
    If planning on using PCA, you must pass a tuple of format(eigen_values, eigen_vectors)
    
    If planning on using Cholesky, you must pass a Correlation matrix C of type np.ndarray 
    
    Use .compute_correlated_noise() to compute correlated noise
    """
    def __init__(self, 
                 n_steps: int=20, 
                 n_simulations: int=1000, 
                 correlation_matrix: np.ndarray=None,
                 eigenvalues: np.ndarray=None,
                 eigenvectors: np.ndarray=None,
                 variable_names: list=None,) -> None:
        
        #Simulation wide params
        self.n_steps = n_steps
        self.n_simulations = n_simulations

        #Required for Cholesky
        self.C = correlation_matrix

        #Required for PCA
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

        #Faculative but useful
        self.variable_names = variable_names
        
        self.decomposition_dict = {"pca" : self._compute_shock_vectors,
                                   "cholesky" : self._correlate_with_cholesky,
                                   "no_transformation" : self._no_transformation
                                   }
    
    @classmethod
    def init_sampler(cls, 
                    n_steps: int=20, 
                    n_simulations: int=1000, 
                    correlation_matrix: np.ndarray=None,
                    eigencomponents: tuple[np.ndarray] | list[np.ndarray]=None,
                    variable_names: list=None,) -> "WhiteNoiseSampler":
        
        if (correlation_matrix is None) & (eigencomponents is None):
            raise ValueError("You must use either correlation_matrix or eigencomponents as components")

        if (eigencomponents is not None):
            if (not len(eigencomponents) == 2):
                raise ValueError("eigen_components must be a tuple/list of format (eigen_values, eigen_vectors)")
            eigenvalues, eigenvectors = eigencomponents
            if (not eigenvalues.shape[0] == eigenvectors.shape[1]):
                raise ValueError("Number of eigenvalues vector should equal to the number of columns of the eigenvector matrix")
    
        if (variable_names is not None):
            if (correlation_matrix is not None):
                if (not len(correlation_matrix) == len(variable_names)):
                    raise ValueError("Variable names must have the same length as correlation matrix")
            elif (eigenvectors is not None):
                if (not len(eigenvectors) == len(variable_names)):
                    raise ValueError("Variable names must have the same length as the eigenvectors")
            
        return WhiteNoiseSampler(n_steps=n_steps, 
                                 n_simulations=n_simulations, 
                                 correlation_matrix=correlation_matrix,
                                 eigenvalues=eigenvalues, 
                                 eigenvectors=eigenvectors,
                                 variable_names=variable_names
                                 )

    @staticmethod
    def _sample_white_noises(n_steps : int, n_factors : int, random_seed: int=None) -> np.ndarray:
        """
        Creates a matrix of white noises with dimensions (n_steps, n_factors)

        Parameters
        ----------
        n_steps : int
            The number of steps in the future we are forecasting
        n_factors : int
            Number of factors in the model, if using PCA should be equal to the number of 
            eigen values kept

        Returns
        -------
        np.array
            The matrix of white noises
        """

        # rows are observations, columns are variables
        #We instantiate a generator with the random_seed and create the white noise with it
        rng = np.random.default_rng(random_seed)
        white_noises = rng.normal(size=(n_steps, n_factors))
        
        return white_noises
        
    @staticmethod
    def _compute_shock_vectors(eigenvalues : np.ndarray, 
                               eigenvectors : np.ndarray, 
                               white_noises : np.ndarray,
                               C: np.ndarray=None,) -> np.ndarray:
        """
        Computes shock vectors based on Pexapark's methodology and the work of Chambard (2024).
        
        Parameters
        ----------
        eigenvalues : np.ndarray
            The eigenvalues kept from the PCA decomposition, with dimensions (Nb of eigenvalues, 1)
        eigenvectors : np.ndarray
            The corresponding eigenvectors from the PCA decomposition, with dimensions (Nb factors, Nb of eigenvalues)
        white_noises : np.ndarray
            A matrix of noise N(0,1) with dimension (N_steps, Nb of eigenvalues)
        C : np.ndarray
            Not used
        
        Returns
        -------
        np.ndarray
            An array of correlated noise with dimensions (Nb factors, Time)
        """
        
        P = eigenvectors #Dims (N_factors, N_eigenvalues) 
        D = np.diag(eigenvalues ** 0.5) #Dims (N_eigenvalues, N_eigenvalues), diag filled with sqrt, all other values 0
        N = white_noises #Dims (T, N_eigenvalues)

        correlated_noises = P @ D @ N.T #(N_factors, N_eigenvalues) @ (N_eigenvalues, N_eigenvalues) @ (N_eigenvalues, T,) -> (N_factors, T)
        
        #NOTE Check the speed with the transpose -> Put the T in rows
        return correlated_noises.T #(N_factors, T) -> (T, N_factors)

    @staticmethod
    def _correlate_with_cholesky(C : np.ndarray, 
                                 white_noises : np.ndarray,
                                 eigenvalues : np.ndarray=None, 
                                 eigenvectors : np.ndarray=None,) -> np.ndarray:
        """
        Cholesky matrix L (triangular lower) is traditionnaly multiplied by 
        a matrix X whose rows are the variables id and columns the observation id.
        
        Therefore, in LX matrix multiplication, each sample is multiplied successively by the Cholesky matrix.
        
        Here we want to have observations in rows and variables in columns : we therefore want transpose(LX) = transpose(X)transpose(L)

        Parameters
        ----------
        white_noises : np.ndarray
            shape is nrows=N_steps, ncols=N_factors
        correlation_matrix : np.ndarray
            shape is (N_factors, N_factors) corresponding to the number of variables to be correlated

        Returns
        -------
        np.ndarray 
            the correlated white_noises with rows as observations and columns as variables
        """
                
        cholesky_matrix = np.linalg.cholesky(C)
    
        correlated_noises = white_noises @ cholesky_matrix.T
        
        return correlated_noises

    @staticmethod
    def _no_transformation(white_noises : np.ndarray,
                           C : np.ndarray=None, 
                           eigenvalues : np.ndarray=None, 
                           eigenvectors : np.ndarray=None,) -> np.ndarray:
        """
        Convenience function that does nothing
        
        Use to allow for an efficient pipeline

        Parameters
        ----------
        white_noises : np.ndarray
            _description_
        C : np.ndarray, optional
            not used, by default None
        eigenvalues : np.ndarray, optional
             not used, by default None
        eigenvectors : np.ndarray, optional
             not used, by default None

        Returns
        -------
        np.ndarray
            Same array as white_noises
        """
        
        return white_noises

    def compute_correlated_noise(self,
                                 method: str="simple",
                                 C: np.ndarray=None,
                                 eigen_components: tuple[np.ndarray, np.ndarray]=None,
                                 n_factors : int=None,
                                 random_seed: int=None) -> tuple[np.ndarray]:
        """
        Run the whole sampler for white noise
        
        Returns white noise or correlated noise depending on whether the object 
        was instantiated with a correlation matrix

        Parameters
        ----------
        C : np.ndarray, optional,
            Correlation matrix, by default None. Only needed if method = "cholesky" and no C eigen_components 
            was passed when instantiating the class
        eigen_components: tuple[np.ndarray, np.ndarray]:
            eigen values and eigen vectors, by default None.
            Passed as a tuple of format (eigen_values, eigen_vectors)
            Only needed if using method = "pca" and no eigen_components argument was passed when
            instantiating the class
        n_factors : int, optional
            Number of factors for which to generate noise, by default 5
            Only needs to be passed if method is None
            n_factors will equal to the number of eigenvalues kept if using PCA method
            if using cholesky, will equal to the dimension of the correlation matrix
        method : str, optional
            What method to use for creating correlated noise, by default None
            If None, no correlation is used, and white noise is returned
            If pca, use the PCA approach to correlate noise -> requires instantiating with 
            eigen_components or passing the eigen_components argument
            If cholesky, use Cholesky Decomposition to correlate noise -> requires instantiating with 
            correlation matrix C or passing the matrix C argument

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple of format (correlated_noises, run_ids)
            run_ids has shape (N_steps, 1, N_simulations)
            correlated_noises has shape (N_steps? N_factors, N_simulations)

        Raises
        ------
        ValueError
            Raised if the argument method is not "pca" or "cholesky" or "simple"
            
        Notes
        -------
        Fortran reshape reshape the index by filling the data column by column
        
        |1,2,3|
        |4,5,6|
        |7,8,9|
        |10,11,12|
        
        reshape to (3,2,2)
        
        |1,4| for z=0 reshaping in F
        |2,5|
        |3,6|

        """
        if (method == "cholesky"): 
            if (C is None):
                if (self.C is None):
                    raise ValueError("No correlation matrix found, please use the C argument or instantiate class with the C argument")
                else: 
                    C = self.C.copy()
            n_factors = len(C)

        elif (method == "pca"):
            if (eigen_components is None):
                if (self.eigenvalues is None) | (self.eigenvectors is None):
                    raise ValueError("No eigenvalues or eigenvectors found, use the eigen_components argument or instantiate class with the eigen_components argument")
                else: 
                    eigenvalues, eigenvectors = self.eigenvalues.copy(), self.eigenvectors.copy()
            else:
                eigenvalues, eigenvectors = eigen_components

            n_factors = len(eigenvalues)

        elif (method == "simple") & (n_factors is None):
            raise ValueError("If no method is passed, you must specify a number of factors to generate")

        decomposition_func = self.decomposition_dict.get(method, None) #Get the correct function based on the method argument
        if (decomposition_func is None): #If an incorrect argument is passed as key, will trigger the KeyError
            raise KeyError("method must be 'pca', 'cholesky', or 'simple'") #NOTE: Call it skip correlation
        
        white_noises = self._sample_white_noises(self.n_steps * self.n_simulations, n_factors, random_seed=random_seed)
        
        params_decomposition = {"C" : C, 
                                "white_noises" : white_noises, 
                                "eigenvalues" : eigenvalues, 
                                "eigenvectors" : eigenvectors
                                }
        correlated_noises = decomposition_func(**params_decomposition)
      
        #Creates an artificial dim for axis 1 so that run_ids has the same dims as white_noises. 
        # Broadcast the run ids over the matrix of 0. Each column will be filled with one number of run id.
        run_ids = (np.zeros((self.n_steps, self.n_simulations)) + np.arange(1, self.n_simulations+1))[:,None,:]
        #Reshape to have 3 dims (n_steps, n_features, n_simulations)
        
        #NOTE : Add a comment
        correlated_noises = correlated_noises.reshape((self.n_steps, self.n_simulations, correlated_noises.shape[1]), order='F').swapaxes(1, 2) 

        return (correlated_noises, run_ids)
    