from pandas import DataFrame, read_csv
from typing import Set
from config.configuration import CATEGORY_TO_NOISE_TIMESTEPS, DEFAULT_RESULTS_PATH


CATEGORIES = set(CATEGORY_TO_NOISE_TIMESTEPS.keys())  # since keys() returns a view and not a set


class ResultsManager:
    """
    A logging class managing loading, updating and storing the experiments results.
    """
    path: str
    is_loaded: bool
    results: DataFrame

    # SUGGESTION:
    # TODO: perhaps we want to make results a persistent @property such that:
    #       'get'-ing the value will read the results file and
    #       'set'-ing the value will update the results file.
    #       
    #       it will be really cool

    def __init__(self, path: str=DEFAULT_RESULTS_PATH):
        self.path = path
        self.is_loaded = False

    def _get_categories_in_results_file(self) -> Set:
        if not self.is_loaded:
            self._load_results()
        
        # TODO: implement
        # 1) Extract a set of categories from self.results
        # 2) return the extracted set
        pass

    def _load_results(self):
        # TODO: implement:
        # read the file
        self.results = ()  # and fill this member
        self.is_loaded = True

    def get_remaining_categories(self) -> Set:
        """
        Getting a set of the remaining categories that do not yet exist 
        in the results file.

        Return:
        -------
        A set of strings denoting the categories who don't exist in the file
        """
        categories_in_file = self._get_categories_in_results_file()

        return CATEGORIES - categories_in_file
    
    def update_results_file(self, new_results: DataFrame) -> None:
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

        # TODO: implement
        pass
