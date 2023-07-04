import os
from typing import List
from pandas import DataFrame, read_csv
from config.configuration import DEFAULT_CSV_DATA_PATH, DEFAULT_RESULTS_COLUMNS


class PersistentDataFrame:
    """
    A logging class managing loading, updating and storing the experiments data.

    Usage:
    ------

    Init:
    >>> persistent_df = PersistentDataFrame(csv_path)

    Getting the current DataFrame stored in the csv and in memory:
    >>> df = persistent_df.data

    Updating the DataFrame presistently:
    >>> persistent_df.data = new_df
    """

    _data_df: DataFrame

    path: str
    is_loaded: bool
    columns: List[str]
    data: DataFrame

    @property
    def data(self) -> DataFrame:
        if not self.is_loaded:
            self._load_data()
        
        return self._data_df
    
    @data.setter
    def data(self, value: DataFrame):
        self._data_df = value.copy(deep=True)
        self._update_data_file(self._data_df)

    def __init__(self, data_path: str=DEFAULT_CSV_DATA_PATH, columns: List[str]=None):
        if not columns:
            columns = DEFAULT_RESULTS_COLUMNS

        self.path = data_path
        self.is_loaded = False
        self.columns = columns

    def _load_data(self):
        """
        Loading the data from the csv at self.path into memory.
        Accessible from self.data
        
        Return:
        -------
        None
        """
        if os.path.exists(self.path):
            self._data_df = read_csv(self.path)
            self.is_loaded = True
        else:
            # Note: Using the setter of the property also creates a file
            self.data = DataFrame(columns=self.columns)

    def _update_data_file(self, new_data: DataFrame) -> None:
        """
        Updates the data csv file with new data by first loading the data into
        memory, updating them using the given newer data, and then saving them back.

        Parameters:
        -----------
        new_data : DataFrame
            The new data we want to update out file with.

        Return:
        -------
        None
        """
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        new_data.to_csv(self.path, index=False)

    def reload_data(self):
        """
        Force reloads the data from the file.
        
        Return:
        -------
        None
        """
        self._load_data()
