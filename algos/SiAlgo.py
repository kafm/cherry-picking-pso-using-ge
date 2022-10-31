from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional, List
import sys, math
import numpy as np
from utils import FitnessFunction, Bounds




EvaluationCallback = Callable[[], None]
GenerationCallback = Callable[[], None]


class SiAlgo(ABC):

    def __init__(
        self, 
        fitnessFunction:FitnessFunction, 
        populationSize:int=20, 
        numGenerations:Optional[int]=None,
        maxEvaluations:Optional[int]=None,
        options: Optional[dict] = None,
        evaluationCallback:Optional[EvaluationCallback]=None,
        generationCallback:Optional[GenerationCallback]=None
    ):
        self.fitnessFunction:FitnessFunction = fitnessFunction
        options = options if options else {} 
        if "populationSize" in options: 
            self.populationSize = options["populationSize"]
        else:
            self.populationSize:int = populationSize
        if numGenerations == None:
            numGenerations = 100 if maxEvaluations == None else 2147483647 #max int
        self.numGenerations:int = numGenerations
        self.maxEvaluations:int = maxEvaluations
        self.evaluationCallback:EvaluationCallback = evaluationCallback
        self.generationCallback:GenerationCallback = generationCallback
        self.evaluations:List[float] = []
        self.options:dict = options
        self.minFit: float = None
         
    @abstractmethod
    def search(self) -> SiSearchResult:
       pass

    def resetEvaluations(self):
        self.evaluations: List[float] = []
        self.minFit = None

    def appendAndAssertEvaluation(self, fitness: float)-> List[float]: 
        self.minFit = fitness if self.minFit == None else min(fitness, self.minFit)
        self.evaluations.append(self.minFit)
        self.assertMaxEvaluationsReached(len(self.evaluations))

    def assertMaxEvaluationsReached(self, numEvals:int):
        if self.maxEvaluations and numEvals > self.maxEvaluations:
            #print("Max evaluations of {} reached".format(self.maxEvaluations))
            raise MaxEvaluationReached("Max evaluations of {} reached".format(self.maxEvaluations))


class SiAgent(ABC):
     def __init__(self, fitnessFunction:FitnessFunction):
         bounds:Bounds = fitnessFunction.bounds()
         self.minval:float = bounds.min
         self.maxval:float = bounds.max
         self.ndims:int = fitnessFunction.dimensionsLength()
         self.fitness:float = sys.float_info.max
         self.pos:List[float] = None  
         self.fitnessFunction:FitnessFunction = fitnessFunction     

class SiSearchResult: 
    def __init__(self,  
        best: SiAgent,
        population: List[SiAgent], 
        fitnessByGeneration: List[float],
        fitnessByEvaluation: List[float]
    ):
        self.best: SiAgent = best
        self.population:List[SiAgent] = population
        self.fitnessByGeneration:List[float] = fitnessByGeneration
        self.fitnessByEvaluation:List[float] = fitnessByEvaluation

class MaxEvaluationReached(Exception):
    pass