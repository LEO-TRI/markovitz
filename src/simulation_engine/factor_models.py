import numpy as np

# use only numpy and list

def force_positivity(vector : np.ndarray) -> np.ndarray:
    """
    helper function to set negative values to zero
    """
    vector[vector < 0] = 0
    return vector


class GeometricBrownianMotion2factor():

    def __init__(self, 
                 sigma1 : float, 
                 sigma2 : float, 
                 tau : float, 
                 rho : float, #Not used, here for compatibility reasons
                 expected_forward_curve : np.ndarray, 
                 sigma_residual: float | np.ndarray=0, #0 since we do * e^sigma_residual so * 1, no shock
                 dt: float=1/12) -> None:
        """
        GeometricBrownianMotion with 2 factor volatility 
        This class assumes correlation between sigma1 and sigma2 is directly given by the correlated noises
        We followed the paper "A Two-Factor Model for the Electricity Forward Market", August 20, 2007
        from Rudiger Kiesel (Ulm University), Gero Schindlmayr (EnBW Trading GmbH), Reik H. B¨orger (Ulm University)

        Parameters
        ----------
        expected_forward_curve : np.array
            each element is the expected price at maturity
        sigma1 : float
            short term volatility
        sigma2 : float
            long term volatility
        tau : float
            decay factor of short term volatility
        rho : float
            here for compatability reasons when unloading dict, not actually used
        dt : timestep, optional
            timestep in the discretization used, by default 1/12
            1/12 for monthly, assuming other parameters are set to annual values
        """
        self.expected_forward_curve = expected_forward_curve #(126,)
        self.sigma_residual = sigma_residual
        self.sigma1 = sigma1 #Scalar
        self.sigma2 = sigma2 #Scalar
        self.tau = tau #Scalar
        self.dt = dt #Scalar
        self.sigma_residual = sigma_residual #(126,)
 
    @staticmethod
    def instant_volatility_function(sigma1 : float, sigma2 : float, t : int, T : int, tau : float, rho: float=0) -> float:
        """Used to represent the volatility function only, T in years since no dt is applied here"""

        return np.sqrt(
            (np.exp(-2 * tau * (T - t)) * sigma1 ** 2)
                + (2 * rho * sigma1 * sigma2 * np.exp(-1 * tau * (T - t))) 
                + (sigma2 ** 2)
                )

    @staticmethod
    def deterministic_volatity_function(sigma1 : float, sigma2 : float, T : int, tau : float, N_steps : int) -> np.ndarray:
        """
        Characteristics of the volatily function 
        decreases with large T
        increases as t gets closer to T

        this function does not take t (the valuation date) as input as t range is built within this function

        Parameters
        ----------
        sigma1 : float
            short term volatility parameter
        sigma2 : float
            long term volatility parameter
        T : int
            product maturity being sampled, in month
        tau : float
            decay factor of short term volatility
        N_steps : int
            maximum horizon at which we sample, in month

        Returns
        -------
        np.array
            The volatility factor matrix with dims (N_steps, 2)
            2 columns (1 for each volatility factor)
        """
        
        if (T <= N_steps) is not True: 
            raise ValueError("Can not sample for T higher than max horizon")
            
        # product with maturity T is sampled until T
        #T_vector = np.repeat(T, repeats=T) #np.ndarray
        # time will go from 1 to T
        t_vector = np.arange(1, T+1, 1) #np.ndarray
        short_term_factor = np.exp(-1 * tau * (T - t_vector)) * sigma1 #np.ndarray, T - t_vector is broadcasted to a 1D array
        
        long_term_factor = np.full(shape=T, fill_value=sigma2, dtype=float) #np.ndarray, marginally faster
        #long_term_factor = np.zeros(T) + sigma2 #np.ndarray, broadcast sigma2 to an 1D array of len T NOTE: ALTERNATIVE
        #long_term_factor = np.repeat(sigma2, repeats=T) #np.ndarray NOTE: ALTERNATIVE
        

        # after T and until N_steps, we add 0 as vol term #Calculate full length NOTE : Calculate 1 full len V and then slice instead
        short_term_factor = np.concatenate([short_term_factor, np.zeros(N_steps - T)]) #np.ndarray with shape (N_steps,)
        long_term_factor = np.concatenate([long_term_factor, np.zeros(N_steps - T)]) #np.ndarray with shape (N_steps,)
        return np.column_stack((short_term_factor, long_term_factor)) #np.ndarray with shape (N_steps, 2)
    
    def sample_forward_curve(self, 
                             sigma_noise_asset : np.ndarray, 
                             residual_noise_asset : np.ndarray,
                             #mu : np.ndarray,
                             N_steps : int) -> np.ndarray:
        """
        Sample 1 forward curve : each forward is sampled until its maturity
        This method is meant to be accessed out of the class, inputing noises and number of steps
        
        The formula to calculate price is:
            - p_i = F_i  * e^(mu_i) * e^(sigma_i * z_i) = F_i  * e^(sigma_i * z_i + mu_i)

        Parameters
        ----------
        sigma_noise_asset : np.ndarray
            The noise for the volatility
        residual_noise_asset : np.ndarray
            The noise for the residuals
        mu : np.ndarray
            The mean  log return value for a month
        N_steps : int
            The number of steps for the simulations

        Returns
        -------
        np.ndarray
            The produced forward curve
        """
        
        forward_curve = []
        forwards_trajectories = []
        # simulations starts here, later this noises would need to be correlated to other markets
        # Pexapark has correlation matrix for sigma1 and sigma2 accross geographies

        #NOTE : Compute 1 full V matrix and then slice iteratively
       
        #OPTION 1
        #volatility_factors = self.deterministic_volatity_function(self.sigma1, self.sigma2, N_steps, self.tau, N_steps) #(N_steps, 2) full, no 0
        #noises_dt = correlated_noises * (self.dt ** 0.5) #NOTE This one needs to be remove from the loop
        #for T in range(1, N_steps+1):
        #   volatility_factors_T = volatility_factors[:T,:]
        #   noises_dt_T = noises_dt[:T,:] 
        #   sigma_dW = np.diag(volatility_factors_T @ noises_dt_T.T)
        #   F_0_T = self.expected_forward_curve[T-1]
        #   forward_at_maturity = F_0_T * np.prod(np.exp(sigma_dW))
        #   forward_curve.append(forward_at_maturity)

        #ALTERNATIVE : Do an iterative descent with recursive slice
        #for T in range(N_steps, 0):
        #   volatility_factors = volatility_factors[:T,:]
        #   noises_dt = noises_dt[:T,:] 
        #   sigma_dW = np.diag(volatility_factors @ noises_dt.T)
        #   F_0_T = self.expected_forward_curve[T-1]
        #   forward_at_maturity = F_0_T * np.prod(np.exp(sigma_dW))
        #   forward_curve.append(forward_at_maturity)
        #forward_curve = forward_curve[::-1]
        
        residual_noise_dt = residual_noise_asset * (self.dt ** 0.5)
        noises_dt = sigma_noise_asset * (self.dt ** 0.5)

        # we loop over different maturities
        for T in range(1, N_steps+1): 

            volatility_factors = self.deterministic_volatity_function(self.sigma1, self.sigma2, T, self.tau, N_steps) #(N_steps, 2)
        
            # time factor is different from one maturity forward to another, but noises are the same
            #NOTE Slice noises_dt here? -> noises_dt = correlated_noises[:T,:] * (self.dt ** 0.5)
            #volatility_factors = volatility_factors[:T,:]

            # page 9 of the paper : the deterministic vol component is multiplied by the stochastics part
            # to obtain the volatility

            sigma_dW = np.diag(volatility_factors @ noises_dt.T) #we only need i,i dot product #np.ndarray

            # expected curve series indexed from 0 to N_steps
            F_0_T = self.expected_forward_curve[T-1] #Scalar, stops 
            #Once the prediction target T is reached, sigma_dW becomes 0 after, e^0 becomes 1 and the product stops moving
            #Potential source of inefficiency since we repeat calculations we don't need -> NOTE : Potentially useful to slice sigma_dW? 
            
            forward_trajectory = F_0_T * np.prod(np.exp(sigma_dW)) #NOTE Switch for np.prod eventually
            
            # last element is the last price. In our vectors, price do not move after maturity (deterministic volatility function set to 0)
            forward_curve.append(forward_trajectory)

        #NOTE pi = Fi  * e^(mu_i) * e^(sigma_i * z_i) = F_i  * e^(sigma_i * z_i + mu_i)

        forward_curve_shocked = (np.asarray(forward_curve) #Forward curve 
                                 * np.exp((residual_noise_dt * self.sigma_residual))# + mu) #shock term here combined with mu
                                 )

        return forward_curve_shocked


class GaussianSampler():
    def __init__(self, 
                 expected_curve_matrix : np.ndarray, 
                 sigma_curve_matrix : np.ndarray, 
                 enforce_positivity: bool=True) -> None:
        """
        Gaussian variables sampler, we assume correlation is already considered in the noises
        
        Can receive data for all assets or only one at a time
        
        Noise data should have 1 more dim than the expected curve and sigma matrixes

        Parameters
        ----------
        expected_curve_matrix : np.array
            expectation at each time step, dims (N_steps, N_factors)
        sigma_curve_matrix : np.array
            standard deviation at each time step, should be provided in absolute value and not in %
            dims (N_steps, N_factors)
        enforce_positivity : bool, optional
            if True the negative values sampled will be set to O, by default True
        """        
        if (expected_curve_matrix.shape == sigma_curve_matrix.shape) is not True:
            raise ValueError("expectation and sigma curves should have same dimensions")
            
        self.expected_curve_matrix = expected_curve_matrix #(N_steps,N_factors)
        self.sigma_curve_matrix = sigma_curve_matrix #(N_steps,N_factors)

        if (enforce_positivity is not True): 
            self.postprocess = lambda x : x        
        else:
            self.postprocess = force_positivity
    
    def sample_curve(self, correlated_noise : np.ndarray) -> np.ndarray:
        """
        if X ~ N(0,1) , then X * sigma + u follows N(u, sigma)

        Parameters
        ----------
        correlated_noise : np.array, (N_steps,N_factors, N_simulations)
            noise vector whose shape should equal the expected_curve

        Returns
        -------
        np.array
            the gaussian vector with law N(expected_curve, sigma_curve)
        """
        #if ((correlated_noise.shape[:2] == self.sigma_curve_matrix.shape) & (correlated_noise.shape[:2] == self.expected_curve_matrix.shape)) is not True:
        #    raise ValueError("Noise should have the same number of cols and rows as the sigma matrix and the mu matrix")

        #TODO: Add a check for unidimensional data ->  1 asset, 1 run
        
        curve = (correlated_noise * self.sigma_curve_matrix[...,None]) + self.expected_curve_matrix[...,None]

        # enforce positivity
        return self.postprocess(curve)
    
    @classmethod
    def compute_on_the_fly(cls, 
                           expected_curve_matrix : np.ndarray, 
                           sigma_curve_matrix : np.ndarray, 
                           correlated_noise : np.ndarray,
                           enforce_positivity: bool=True) -> np.ndarray:
        """
        Convenience function to compute the sample curve on the fly without keeping in memory
        an instantiated version of the GaussianSampler 

        Parameters
        ----------
        expected_curve_matrix : np.array
            expectation at each time step, dims (N_steps, N_factors)
        sigma_curve_matrix : np.array
            standard deviation at each time step, should be provided in absolute value and not in %
            dims (N_steps, N_factors)
        correlated_noise : np.array, (N_steps,N_factors, N_simulations)
            noise vector whose shape should equal the expected_curve
        enforce_positivity : bool, optional
            if True the negative values sampled will be set to O, by default True

        Returns
        -------
        np.array
            the gaussian vector with law N(expected_curve, sigma_curve)
        """
        
        return GaussianSampler(expected_curve_matrix=expected_curve_matrix, 
                               sigma_curve_matrix=sigma_curve_matrix,
                               enforce_positivity=enforce_positivity).sample_curve(correlated_noise)


# ************** for each factor we assign a model   

PriceSampler = GeometricBrownianMotion2factor
CaptureRatioSampler = GaussianSampler
VolumeSampler = GaussianSampler


