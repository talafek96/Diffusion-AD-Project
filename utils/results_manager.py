import os
from typing import Set
from pandas import DataFrame, read_csv
from config.configuration import CATEGORY_TO_NOISE_TIMESTEPS, DEFAULT_RESULTS_PATH


ALL_CATEGORIES = set(CATEGORY_TO_NOISE_TIMESTEPS.keys())  # since keys() returns a view and not a set


class ResultsManager:
    """
    A logging class managing loading, updating and storing the experiments results.

    Usage:
    ------

    Init:
    >>> results_manager = ResultsManager(csv_path)

    Getting the current results DataFrame:
    >>> results = results_manager.results

    Updating the results DataFrame presistently:
    >>> results_manager.results = new_results
    """
    _results_df: DataFrame

    path: str
    is_loaded: bool

    @property
    def results(self) -> DataFrame:
        if not self.is_loaded:
            self._load_results()
        
        return self._results_df
    
    @results.setter
    def results(self, value: DataFrame):
        self._results_df = value.copy(deep=True)  # TODO: maybe we don't need a deep copy or a copy at all
        self._update_results_file(self._results_df)


    def __init__(self, results_path: str=DEFAULT_RESULTS_PATH):
        self.path = results_path
        self.is_loaded = False

    def _load_results(self):
        """
        Loading the results from the csv at self.path into memory.
        Accessible from self.results
        
        Return:
        -------
        None
        """
        if os.path.exists(self.path):
            self._results_df = read_csv(self.path)
            self.is_loaded = True
        else:
            # Note: Using the setter of the property also creates a file
            self.results = DataFrame(columns=["category", "img_auc", "pixel_auc"])

    def _update_results_file(self, new_results: DataFrame) -> None:
        """
        Updates the results csv file with new results by first loading the results into
        memory, updating them using the given newer results, and then saving them back.

        Parameters:
        -----------
        new_results : DataFrame
            The new results we want to update out file with.

        Return:
        -------
        None
        """
        new_results.to_csv(self.path)

    def _get_categories_in_results_file(self) -> Set:
        """
        Getting a list of the categories from the results file.
        
        Return: Set[str]
        -------
        A set of strings denoting the categories stored in the results file.
        """
        if not self.is_loaded:
            self._load_results()
        
        categories = set(self._results_df["category"])
        
        return categories

    def reload_results(self):
        """
        Force reloads the results from the file.
        
        Return:
        -------
        None
        """
        self._load_results()

    def get_remaining_categories(self) -> Set:
        """
        Getting a set of the remaining categories that do not yet exist 
        in the results file.

        Return:
        -------
        A set of strings denoting the categories who don't exist in the file
        """
        categories_in_file = self._get_categories_in_results_file()

        return ALL_CATEGORIES - categories_in_file


def test():
    # TODO:
    pass


if __name__ == '__main__':
    test()
