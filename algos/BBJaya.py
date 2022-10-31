from __future__ import annotations
from typing import Optional
from numba import njit
import numpy as np
from utils import FitnessFunction
from .SiAlgo import *


class BBJaya(SiAlgo):
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
        worse: Agent = None
        agents: List[Agent] = [Agent(self.fitnessFunction, reportEval) for i in range(self.populationSize)]
        generations: List[float] = []
        try:
            for _ in range(self.numGenerations):
                best = min(agents, key=lambda a: a.fitness)
                worse = max(agents, key=lambda a: a.fitness)
                for a in agents:
                    a.update(best, worse)
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
        reportEvaluation(self)

    def update(self, best: Agent, worse: Agent):
        self.update1(best, worse)

    def update3(self, best: Agent, worse: Agent):
        bbPart = np.random.normal(getMean(self.pos, best.pos), getSpread(self.pos, best.pos))
        jayaPart = np.random.normal(getMean(self.pos, worse.pos), getSpread(self.pos, worse.pos))
        pos = getNextPos(bbPart - jayaPart, self.minval, self.maxval)
        fit = self.fitnessFunction.evaluate(pos)
        if (fit < self.fitness):
            self.fitness = fit
            self.pos = pos
        self.reportEvaluation(self)

    def update2(self, best: Agent, worse: Agent):
        bbPart = np.random.normal(getMean(self.pos, best.pos), getSpread(self.pos, best.pos))
        jayaPart = np.random.normal(getMean(self.pos, worse.pos), getSpread(self.pos, worse.pos))
        pos = getNextPos(bbPart - jayaPart, self.minval, self.maxval)
        fit = self.fitnessFunction.evaluate(pos)
        if (fit < self.fitness):
            self.fitness = fit
            self.pos = pos
        self.reportEvaluation(self)

    def update1(self, best: Agent, worse: Agent):
        #bestPart = np.random.normal(getMean(self.pos, best.pos), getSpread(self.pos, best.pos))
        #worsePart = np.random.normal(getMean(self.pos, worse.pos), getSpread(self.pos, worse.pos))
        #pos = getNextPos(bestPart, self.minval, self.maxval) - getNextPos(worsePart, self.minval, self.maxval)
        absPos = np.abs(self.pos) 
        bbMeanPart = getMean(self.pos, best.pos)
        jayaMeanPart = (worse.pos - absPos)/2
        bbSpreadPart = getSpread(self.pos, best.pos)
        pos = np.random.normal(bbMeanPart - jayaMeanPart, bbSpreadPart)
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
