from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import sys

class FitnessFunction(ABC):

    def __init__(self, numDimensions:int=2, opposite:bool=False):
        self.numDimensions = numDimensions
        self.opposite = opposite
   
    @abstractmethod
    def evaluate(self, X: np.ndarray) -> float: 
       pass

    def bounds(self) -> Bounds:
        pass
    
    def dimensionsLength(self) -> int:
        return self.numDimensions

class Bounds:
    def __init__(self, min:float=sys.float_info.min, max:float=sys.float_info.max):
        self.min:float = min
        self.max:float = max

    def __str__(self) -> str:
        return "Min: {min}, Max {max}".format(min=self.min, max=self.max)