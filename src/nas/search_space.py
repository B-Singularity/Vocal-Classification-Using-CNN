from abc import ABC, abstractmethod

class SearchSpace:
    def __init__(self):
        pass

    @abstractmethod
    def get_search_space_ops(self) -> dict:
        """
        Returns search space ops as a dictionary of lists.
        """
        pass
