from __future__ import annotations
from typing import Optional
from numba import njit
import numpy as np
from utils import FitnessFunction
from scipy.stats import cauchy
from .SiAlgo import *


class Cauchy(SiAlgo):
    def __init__(self,
                 fitnessFunction: FitnessFunction,
                 populationSize: Optional[int] = None,
                 numGenerations: Optional[int] = None,
                 maxEvaluations: Optional[int] = None,
                 options: Optional[dict] = None,
                 evaluationCallback: Optional[EvaluationCallback] = None,
                 generationCallback: Optional[GenerationCallback] = None
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
        best: Agent = None
        agents: List[Agent] = [Agent(self.fitnessFunction, reportEval) for i in range(self.populationSize)]
        generations: List[float] = []
        try:
            for _ in range(self.numGenerations):
                best = min(agents, key=lambda a: a.fitness)
                for a in agents:
                    a.update(best)
                generations.append(best.fitness)
        except MaxEvaluationReached:
            None
        return SiSearchResult(
            best=best,
            population=agents,
            fitnessByGeneration=generations,
            fitnessByEvaluation=self.evaluations
        )


class Agent(SiAgent):
    def __init__(self, fitnessFunction: FitnessFunction, reportEvaluation: Callable):
        super().__init__(fitnessFunction)
        bounds = fitnessFunction.bounds()
        self.minval: float = bounds.min
        self.maxval: float = bounds.max
        self.ndims: int = fitnessFunction.dimensionsLength()
        self.pos = np.random.uniform(
            low=self.minval, high=self.maxval, size=self.ndims)
        self.fitness = fitnessFunction.evaluate(self.pos)
        self.reportEvaluation: Callable = reportEvaluation
        #self.coolingFactor:float = .279 #sphere
        self.coolingFactor:float = .35
        reportEvaluation(self)

    def update(self, best: Agent):
        pos = cauchy.rvs(loc=getMean(self.pos, best.pos), scale=getSpread(self.pos, best.pos)*self.coolingFactor, size=self.ndims) 
        pos = getNextPos(pos, self.minval, self.maxval)
        fit = self.fitnessFunction.evaluate(pos)
        if (fit < self.fitness):
            self.fitness = fit
            self.pos = pos
        self.reportEvaluation(self)


@njit(nogil=True)
def getMean(currPos: np.ndarray, otherPos: np.ndarray) -> np.ndarray:
    return np.add(currPos, otherPos) / 2

@njit(nogil=True)
def getSpread(currPos: np.ndarray, otherPos: np.ndarray) -> np.ndarray: #standard deviation
    return np.abs(np.subtract(currPos, otherPos))

@njit(nogil=True)
def getNextPos(pos: np.ndarray, minval: float, maxval: float) -> np.ndarray:
    return np.clip(pos, minval, maxval)
