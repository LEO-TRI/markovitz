import numpy as np
import pandas as pd
import sys 

sys.path.append("/Workspace/Repos/OPTA/Databricks")
from opta.src.simulation_engine.factor_models import PriceSampler, CaptureRatioSampler, VolumeSampler

#TODO Compare with Jockey's results
#TODO Filter on only ACEEF results: get the site IDs for ACEEF assets and associated markets


class SimulationManager:
    
    def __init__(self, 
                 N_SIMULATIONS : int,
                 N_STEPS : int,
                 start_date : str,
                 frequency: str="M",
                 zone_list: list=None,
                 asset_list: list=None,
                 dt: int=12) -> None:
                
        #Data holders
        self.zone_list = zone_list if isinstance(zone_list, (list, tuple)) else [zone_list]
        self.asset_list = asset_list if isinstance(asset_list, (list, tuple)) else [asset_list]
        self.run_ids = range(1, N_SIMULATIONS+1)

        #Params
        self.start_date = start_date
        self.frequency = frequency
        self.N_SIMULATIONS = N_SIMULATIONS
        self.N_STEPS = N_STEPS
        self.dt = dt
    
    def __repr__(self) -> str:
        print(f"Sim params: N_SIMULATIONS = {self.N_SIMULATIONS} & N_STEPS = {self.N_STEPS}")
    
    ####################PRICE SIMULATION####################
    def run_price_simulations(self, 
                              inplace: bool=True) -> "SimulationManager":
        """
        Runs all price simulations based on the price_params dict passed when instantiating

        Parameters
        ----------
        sigma_noise : np.ndarray
            An array of noise, where every 2 columns is sigma 1/sigma 2 for a given zone
            Dims (N_steps, 2 * N_zones, N_simulations)
        residual_noise : np.ndarray
            An array of noise, where every 1 column is residual noise for a given zone
            Dims (N_steps, N_zones, N_simulations)
        inplace : bool, optional
            Whether to return the updated, by default True
            If false, returns the results in a dict format

        Returns
        -------
        SimulationManager
            Instance of the simulation manager
        """
        
        results = []
        
        #Same for each zone simulation
        
        for (i, zone) in enumerate(self.zone_list): 

            #zone_results = {} #Contains the results for 1 zone
            
            prices_results = self._run_one_price_simulation(params_simulation=zone.price_params,
                                                            sigma_noise=zone.sigma_noise, 
                                                            residual_noise=zone.residual_noise #Removes a potential issue where the array has 1 too many dims
                                                            )
            #Stores the price simulations
            results.append(zone.update_asset(new_param={"price_results": prices_results
                                                        }
                                             )
                           )
        if inplace:
            self.zone_list = results.copy()
            return self

        return results
                    
    def _run_one_price_simulation(self, 
                                  params_simulation : dict, 
                                  sigma_noise : np.ndarray,
                                  residual_noise: np.ndarray) -> np.ndarray:
        """
        Runs the price simulation for one zone
        
        Should usually be used only within run_price_simulations()

        Parameters
        ----------
        params_simulation : dict
            The simulation parameters in a dict format. Should include:
                Sigma 1
                Sigma 2
                Tau 
                Expected forward curve
        sigma_noise : np.ndarray
            A (N_steps, 2, N_simulations) array with noise respectively for sigma 1 and sigma 2
        residual_noise : np.ndarray
            The residual noise, with format (N_steps, N_simulations)

        Returns
        -------
        np.ndarray
            Price simulations with format (N_steps, N_simulations)
        """
            
        price_model = PriceSampler(**params_simulation, #unload one dict per market
                                   dt=self.dt,
                                   sigma_residual=0 #NOTE: Eventually integrate sigma_residual in the dict
                                    )
        
        prices_results = np.empty((self.N_STEPS, self.N_SIMULATIONS)) #Generate an empty array we will fill iteratively with the simulations
        
        #Each column is one simulation, each row is one time step. So default would be dims (126, 10**4)
        #Noise has the simulation dimension built on axis 2 (3rd dim), so each simulation we slice a different index on axis 2
        #WARNING: The 3rd dim of noise becomes the 2nd dim of prices_results
        for i in range(self.N_SIMULATIONS): 

            prices_results[:,i] = price_model.sample_forward_curve(sigma_noise_asset=sigma_noise[...,i], 
                                                                   residual_noise_asset=residual_noise[...,i], 
                                                                   N_steps=self.N_STEPS
                                                                   )
            
        return prices_results
    
    ####################VOLUME SIMULATION####################
    def run_volume_simulations(self, 
                               inplace: bool=False,
                               enforce_positivity: bool=True) -> "SimulationManager":
        """
        Run asset simulations for all assets in the volume_params

        Parameters
        ----------
        volume_noise : np.ndarray
            An array of volume noise with dims (N_steps, N_assets, N_simulations)
        inplace : bool, optional
            Whether to return the updated, by default True
            If false, returns the results in a dict format
        enforce_positivity: bool, optional
            If true, forces negative values to 0, by default True

        Returns
        -------
        SimulationManager
            An updated version of the SimulationManager
        """
        
        n_cols = len(self.asset_list) 
        results = []
        
        #We vectorize all the operations for volume calculation. In consequence, we need to
        #extract the arrays in the dict and put them in a matrix. 
        #nth column for those arrays correspond to the array for the nth asset in the volume_params dict

        #2D arrays of shape (n_steps, n_factors) and 1 3D arrays (n_steps, n_factors, n_simulations)
        #Same process as for the volume. NOTE: Should really fuse the two processes.
        mean_volume_array = np.empty((self.N_STEPS, n_cols))
        sigma_volume_array = np.empty((self.N_STEPS, n_cols))
        volume_noise = np.empty((self.N_STEPS, n_cols, self.N_SIMULATIONS))
        
        for (i, asset) in enumerate(self.asset_list):
            mean_volume_array[:,i] = asset.volume_params["expected_curve"]
            sigma_volume_array[:,i] = asset.volume_params["sigma_curve"]

            #3D arrays of shape (n_steps, n_factors, n_simulations). Slice if need to do less simulations than noise was generated
            volume_noise[:,i,:] = asset.volume_noise[...,:self.N_SIMULATIONS]
        
        #Volume noise is (N_steps, N_assets, N_simulations), sigma and mean arrays are (N_steps, N_assets)
        #We expand their dims to (N_steps, N_assets, 1) to allow for broadcasting the operations. Final results
        #has dimensions (N_steps, N_assets, N_simulations)

        volume_results = VolumeSampler.compute_on_the_fly(mean_volume_array, 
                                                          sigma_volume_array, 
                                                          volume_noise, 
                                                          enforce_positivity=enforce_positivity
                                                        )
        #Special method that returns a new instance of the class with the updated params
        #The list is new instances of the class, separated from self.asset_list
        results = [asset.update_asset(new_param = {"volume_results":volume_results[:,i,:],
                                                   "run_ids":self.run_ids,
                                                   }
                                      )
                   for (i, asset) in enumerate(self.asset_list)
                   ]
        if inplace: #TODO Crush the asset list or create a new one?
            self.asset_list = results.copy()
            #self.volume_results = results.copy()
            return self

        return results.copy()
    
    ####################CAPTURE SIMULATION####################
    def run_capture_simulation(self, 
                                inplace: bool=False,
                                enforce_positivity: bool=True) -> "SimulationManager":
        """
        Runs the capture simulations for all keys passed in the capture_params argument

        Parameters
        ----------
        capture_noise : np.ndarray
            The noise for the capture estimations, with dims (N_steps, N_zones, N_simulations)
        inplace : bool, optional
            Whether to return the updated, by default True
            If false, returns the results in a dict format

        Returns
        -------
        SimulationManager
            An updated version of the SimulationManager storing the results for capture simulations 
        """
    
        #TODO : Change process to asset level
        
        n_cols = len(self.asset_list)
        results = []

        #2D arrays of shape (n_steps, n_factors)
        #Same process as for the volume. NOTE: Should really fuse the two processes.
        mean_capture_array = np.empty((self.N_STEPS, n_cols))
        sigma_capture_array = np.empty((self.N_STEPS, n_cols))
        capture_noise = np.empty((self.N_STEPS, n_cols, self.N_SIMULATIONS))
        
        for (i, asset) in enumerate(self.asset_list):
            mean_capture_array[:,i] = asset.capture_params["expected_curve"]
            sigma_capture_array[:,i] = asset.capture_params["sigma_curve"]

            #3D arrays of shape (n_steps, n_factors, n_simulations)
            capture_noise[:,i,:] = asset.capture_noise[...,:self.N_SIMULATIONS]

        #Class method that computes the sample curve on the fly, does not require to have an instantiated class here
        capture_results = CaptureRatioSampler.compute_on_the_fly(expected_curve_matrix=mean_capture_array, 
                                                                 sigma_curve_matrix=sigma_capture_array, 
                                                                 correlated_noise=capture_noise, 
                                                                 enforce_positivity=enforce_positivity
                                                                 )
        
        #Special method that returns a new instance of the class with the updated params
        #The list is new instances of the class, separated from self.asset_list
        results = [asset.update_asset(new_param = {"capture_results":capture_results[:,i,:],
                                                   "run_ids": self.run_ids,
                                                   }
                                      )
                   for (i, asset) in enumerate(self.asset_list)
                   ]
        if inplace: #NOTE Crush the asset list or create a new one?
            self.asset_list = results.copy()
            return self

        return results.copy()    
    
    def format_output(self, asset_list: list=None) -> tuple[pd.DataFrame]:
        
        if (asset_list is None): 
            assets_to_process = self.asset_list.copy() #Select all assets
        else:
            assets_to_process = [asset for asset in self.asset_list if (asset.site_id in asset_list)]
        
        #Build a 2 matrix with cols assets and rows time steps
        #And then flattening to a 1D array column by column -> Goal is to have 1 column per factor with 
        #the asset in index with the run IDs
        #Get the site_id and build an index with each combination of run ID and site_id
        asset_names = (asset.site_id for asset in assets_to_process) 
        steps = pd.date_range(start=self.start_date, periods=self.N_STEPS, freq="M")
        df_asset = pd.DataFrame(index=pd.MultiIndex.from_product(iterables=(asset_names, 
                                                                            self.run_ids, 
                                                                            steps), 
                                                                names=("site_id","runID", "timestamp")
                                                          )
                                )
        df_asset["volume"] = (np.concatenate([asset.volume_results for asset in assets_to_process],
                                             axis=1
                                            )
                                .flatten(order="F")
                                )
        df_asset["capture"] = (np.concatenate([asset.capture_results for asset in assets_to_process],
                                             axis=1
                                            )
                                 .flatten(order="F")
                                )
        market_names = (market.name for market in self.zone_list) 
        df_price = pd.DataFrame(index=pd.MultiIndex.from_product(iterables=(market_names, 
                                                                            self.run_ids, 
                                                                            steps), 
                                                    names=("ardianSpotName","runID", "timestamp")
                                                    )
                                )
        df_price["price"] = (np.concatenate([market.price_results for market in self.zone_list],
                                            axis=1
                                            )
                                .flatten(order="F")
                                )
        
        return (df_asset, df_price)
    
    @classmethod
    def run_full_simulations(cls,
                             N_SIMULATIONS : int,
                            N_STEPS : int,
                            start_date : str,
                            frequency: str="M",
                            zone_list: list=None,
                            asset_list: list=None,
                            dt: int=12) -> "SimulationManager":
        
        return (SimulationManager(N_SIMULATIONS=N_SIMULATIONS,
                                  N_STEPS=N_STEPS,
                                  start_date=start_date,
                                  frequency=frequency,
                                  zone_list=zone_list,
                                  asset_list=asset_list,
                                  dt=dt).run_capture_simulation(inplace=True)
                                        .run_volume_simulations(inplace=True)
                                        .run_price_simulations(inplace=True)
                                        .format_output()
                                   )
        
