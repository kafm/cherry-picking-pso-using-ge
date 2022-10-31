from __future__ import annotations
from typing import Optional
from numba import njit
import numpy as np
from utils import FitnessFunction
from .SiAlgo import *

class BarebonesPSO(SiAlgo):
    def __init__(self, 
        fitnessFunction:FitnessFunction, 
        populationSize:Optional[int]=None, 
        numGenerations:Optional[int]=None,
        maxEvaluations:Optional[int]=None,
        options:Optional[dict]=None,
        evaluationCallback:Optional[EvaluationCallback]=None,
        generationCallback:Optional[GenerationCallback]=None
    ):
        super().__init__(
            fitnessFunction, 
            populationSize=populationSize, 
            numGenerations=numGenerations,
            maxEvaluations=maxEvaluations,
            options=options,
            evaluationCallback=evaluationCallback,
            generationCallback=generationCallback
        )

    def search(self) -> SiSearchResult:
        self.resetEvaluations()
        reportEval = lambda p: self.appendAndAssertEvaluation(p.fitness) 
        best:Particle = None
        particles:List[Particle] = [Particle(self.fitnessFunction, reportEval) for i in range(self.populationSize)]
        generations:List[float] = []
        try:
            for _ in range(self.numGenerations):
                best = min(particles, key = lambda p : p.fitness)
                for p in particles:
                    p.update(best)
                generations.append(best.fitness)
        except MaxEvaluationReached:
            None   
        return SiSearchResult(
            best=best,
            population=particles, 
            fitnessByGeneration=generations,
            fitnessByEvaluation=self.evaluations
        )


class Particle(SiAgent):
    def __init__(self, fitnessFunction:FitnessFunction, reportEvaluation:Callable):
        super().__init__(fitnessFunction)
        bounds = fitnessFunction.bounds()
        self.minval:float = bounds.min
        self.maxval:float = bounds.max
        self.ndims:int = fitnessFunction.dimensionsLength()
        self.pos = np.random.uniform(low=self.minval, high = self.maxval, size = self.ndims)
        self.best:List[float] = self.pos
        self.fitness = fitnessFunction.evaluate(self.pos)
        self.reportEvaluation: Callable = reportEvaluation
        self.coolingFactor:float = 1 #TO TEST IF IT APPLIES. BUILD HYPhOTESIS .85
        reportEvaluation(self)
     
    def update(self, best:Particle):
        #pos = np.random.normal(np.add(self.best, best.best) / 2, abs(np.subtract(self.best, best.best)))
        pos = np.random.normal(getNextPosMean(self.best, best.best), getNextPosSpread(self.best, best.best)*self.coolingFactor)
        pos = getNextPos(pos, self.minval, self.maxval)
        fit = self.fitnessFunction.evaluate(pos)
        if(fit < self.fitness): 
            self.fitness = fit
            self.best = pos
        self.reportEvaluation(self)

@njit(nogil=True)
def getNextPosMean(currPos: np.ndarray, bestPos: np.ndarray) -> np.ndarray:
    return np.add(currPos, bestPos) / 2

@njit(nogil=True)
def getNextPosSpread(currPos: np.ndarray, bestPos: np.ndarray) -> np.ndarray: #standard deviation
    return np.abs(np.subtract(currPos, bestPos))

@njit(nogil=True)
def getNextPos(pos: np.ndarray, minval: float, maxval: float) -> np.ndarray:
    return np.clip(pos, minval, maxval)

