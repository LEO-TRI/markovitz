import numpy as np
import pandas as pd
from dataclasses import dataclass 
from _collections_abc import Callable
import seaborn as sns

@dataclass
class Simulation:
    
    name : str
    
    def update_asset(self, new_param : dict) -> Callable:
        """
        Convenience class to create updated instances of Zone and Asset
        
        Avoids inplace modifications by recreating 

        Parameters
        ----------
        new_param : dict
            The parameter to be updated, must have a format (attribute, value)
            or {attribute : value}

        Returns
        -------
        Callable
            A new instance of the class passed
        """
        
        params = self.__dict__.copy()        
        params.update(new_param) #Inplace updating of the params dict
        
        return type(self)(**params) #Returns a new instance of the class with updated parameters
    
    def show_histogram(self, data_type : str, title: str=None): #TODO: Data has become too big for plotly 

        if (title is None):
            title = f"Distribution of {data_type}"

        if (type(self) == "Zone"):
            if (data_type == "price"):
                data = self.price_results
            else:
                raise ValueError("data_type should be 'price'")
        else:
            if (data_type == "capture"):
                data = self.capture_results
            elif (data_type == "volume"):
                data = self.volume_results
            elif (data_type == "revenue"):
                data = self.revenues_results

        #Histograms are done on all the data. So if arrays are multidimensional, flatten to 1D. 
        #Since data are stored by asset, should already be the case
        if (data.ndim > 1):
            data = data.flatten()

        data_df = pd.DataFrame(data={data_type : data})

        fig = sns.histplot(data_df, x=data_type, title=title) 
        
        return fig
    

@dataclass 
class Zone(Simulation):
    
    #Inputs
    price_params: dict
    residual_noise: np.ndarray=None #1D (N_steps,)
    sigma_noise: np.ndarray=None #2D (N_steps,2)
    
    #Outputs
    run_ids : np.ndarray=None
    price_results: np.ndarray=None #2D, (N_steps, N_simulations)
    

@dataclass    
class Asset(Simulation): #Site
    
    site_id : str
    technology : str=None
    ardianSpotName : str=None #ardianSpotName
    
    #Inputs
    volume_params: dict=None #Dict with 2 keys, value & sigma
    capture_params: dict=None #Dict with 2 keys, value & sigma
    volume_noise: np.ndarray=None #1D (N_steps,)
    capture_noise: np.ndarray=None #1D (N_steps,)
    
    #Outputs
    run_ids : np.ndarray=None
    volume_results: np.ndarray=None #2D, (N_steps, N_simulations)
    capture_results: np.ndarray=None #2D, (N_steps, N_simulations)

    #Calculated outputs: from outputs and price params
    revenues_results: np.ndarray=None


