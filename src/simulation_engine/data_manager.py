import numpy as np
import pandas as pd
import functools as ft

import os
from collections.abc import Callable

class DataManager:
    
    def __init__(self, 
                 load_func : Callable, 
                 save_func : Callable,
                 spark,
                 simulation_ID : int) -> None:
        
        self.load_func = load_func
        self.save_func = save_func
        self.spark = spark
        
        self.simulation_ID = simulation_ID
                    
    @classmethod
    def init_simulation(cls, simulation_ID : str, list_paths : list[str], list_sheet_names : list[tuple], N_STEPS: int=126) -> "DataManager":
        """
        Load data from lake 

        Parameters
        ----------
        list_paths : list[str]
            _description_
        list_sheet_names : list[tuple]
            _description_
        N_STEPS : int, optional
            _description_, by default 126

        Returns
        ----------
        DataManager
            _description_
        """
        
        list_all_data = []
        for file_path, sheet_names in zip(list_paths, list_sheet_names):
            list_loaded_data = []
            for sheet in sheet_names:
                list_loaded_data.append(cls._read_excel_from_lake(file_path, sheet))
            list_all_data.append(tuple(list_loaded_data
                                       )
                                 )
        data_dict = dict(zip(("prices","volume","capture"), list_all_data))
        
        shared_path = Path
        params_path = Path(shared_path) / simulation_ID / "inputs"
        simulation_params = cls._load_params_from_lake(params_path)
        simulation_params = dict(simulation_params) #NOTE provisional
         
        return DataManager(simulation_ID=simulation_ID, N_STEPS=N_STEPS, **data_dict)
            
    def get_prices(self, N_STEPS : int) -> np.ndarray:
        """
        _summary_

        Returns
        -------
        np.ndarray
            _description_
        """
        
        return self.prices.to_numpy()[:N_STEPS]
    
    def get_volume(self, asset_name : str, N_STEPS : int) -> tuple[np.ndarray, np.ndarray]:
        """
        _summary_

        Parameters
        ----------
        asset_name : str
            _description_

        Returns
        -------
        tuple[np.ndarray]
            _description_
        """
                
        volume_df = (pd.Series(pd.to_datetime(self.prices["date"]).dt.month)
                       .to_frame(name ="month")
                       .set_index("month")
                       )
        volume_df = (volume_df.join(self.volume.loc[self.volume["assetName"] == asset_name,["P50Month", "stdMonth"]])
                              .astype(float)
                              )     
        volume_arr = volume_df.to_numpy()[:N_STEPS,:]   
        return (volume_arr[:,0], volume_arr[:,1])
    
    def get_capture(self, market_name : str, technology_name : str) -> tuple[np.ndarray]:
        """
        _summary_

        Parameters
        ----------
        market_name : str
            _description_
        technology_name : str
            _description_

        Returns
        -------
        tuple[np.ndarray]
            _description_
        """
        
        capture_df, capture_params = self.capture
        
        capture_params = (capture_params.set_index(["year", "market", "technology"])
                                        .loc[:, ["value"]]
                                        .rename(columns={"value" : "sigma"})
                                        )
        capture_df = (capture_df.copy()
                            .loc[lambda x: (x["market"] == market_name) & (x["technology"] == technology_name),:]
                            .assign(year = pd.to_datetime(self.capture["date"]).dt.year)
                            .set_index(["year", "market", "technology"])
                            ) 
        capture_df = (capture_df.rename(columns={"value" : "expected"})
                                .join(capture_params, how="left")
                                .reset_index()
                                .loc[:,["expected", "sigma"]]
                                .astype(float)
                                )
        capture_arr = capture_df.to_numpy()   
        return (capture_arr[:,0], capture_arr[:,1])
    
    
    def save_data_from_lake(self,
                            df : pd.DataFrame,
                            base_path : str,
                            asset_type : str,
                            unique_identifier : str,
                            simulation_ID: int=None,
                            save_mode: str="overwrite") -> None:
        """
        Saving the simulations on the lake

        Parameters
        ----------
        base_path : str
            Base path shared by all simulations
        asset_type : str
            Can be a market (if saving prices) or an asset (if saving volumes & captures)
        list_to_save : list
            A list of Assets objects or Zone objects
        """
        if (simulation_ID is None):
            simulation_ID = self.simulation_ID

        data_path = os.path.join(base_path, str(simulation_ID), "output", asset_type)

        save_path = os.path.join(data_path, f"simulation_output.parquet")

        self.save_func(df, save_path, self.spark, save_mode="overwrite")
            
        return None

    def load_params_from_lake(self, 
                              base_path : str, 
                              data_type : str, 
                              assets_aceef : pd.DataFrame=None, 
                              column_names : list[str]=None,) -> pd.DataFrame:
        """
        _summary_
    
        Parameters
        ----------
        base_path : str
            _description_
        data_type : str
            What we are pullin from the lake : can be correlation, prices, volumes or captures
        assets_aceef : pd.DataFrame
            _description_
        column_names : list[str]
            _description_

        Returns
        -------
        pd.DataFrame
            _description_

        Raises
        ------
        TypeError
            _description_
        """
        
        if (data_type == "correlation"):
            
            end_path = "correlation-matrix"
            data_path = os.path.join(base_path, data_type, end_path)
            
            return self.load_func(self.spark, data_path)
        
        #Create a set to get all unique values. If assets_aceef[column_names] is a series, create a set of size 1 tuples. 
        #If assets_aceef[column_names] is a pd.DataFrame, then creates 

        if isinstance(column_names, (str, list)):
            unique_values = set(zip(assets_aceef[column_names].to_numpy())) 
        else:
            raise TypeError("Argument column_names must be a string or a list of string")
                
        result_list = []
                
        for val in unique_values:
            
            #Necessary to rewrite data_path at each iteration
            data_path = os.path.join(base_path, data_type)
            
            if (len(val) > 1):
                #Unload the tuple into a new tuple. Last value is sliced out bc it is used in end_path instead
                data_path = os.path.join(data_path, *val[:-1]) 
 
            end_path = f"{val[-1]}_{data_type}.parquet"
            data_path = os.path.join(data_path, end_path)
                        
            loaded_values = self.load_func(self.spark, data_path.as_posix())
            loaded_values[data_type] = val
            
            result_list.append(loaded_values)
            
        result_df = pd.concat(result_list, axis=0, ignore_index=True)
        
        return result_df
        