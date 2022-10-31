from __future__ import annotations
from typing import Optional
from numba import njit
import copy
import numpy as np
from utils import FitnessFunction
from .SiAlgo import *

class GreyWolf(SiAlgo):
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
        wolfs:List[Wolf] = sorted([Wolf(self.fitnessFunction, reportEval) for i in range(self.populationSize)], key = lambda w: w.fitness)
        alphaWolf, betaWolf, gammaWolf = copy.copy(wolfs[: 3])
        generations:List[float] = []
        try:
            for i in range(self.numGenerations):
                a = 2*(1 - i/self.numGenerations)
                for w in wolfs:
                   w.update(alphaWolf, betaWolf, gammaWolf, a)
                wolfs = sorted(wolfs, key = lambda temp: temp.fitness)
                alphaWolf, betaWolf, gammaWolf = copy.copy(wolfs[: 3])
                generations.append(alphaWolf.fitness)
        except MaxEvaluationReached:
            None   
        return SiSearchResult(
            best=alphaWolf,
            population=wolfs, 
            fitnessByGeneration=generations,
            fitnessByEvaluation=self.evaluations
        )


class Wolf(SiAgent):
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
        reportEvaluation(self)
     
    def update(self, alphaWolf:Wolf, betaWolf:Wolf, gammaWolf:Wolf, a:float):
        A1, A2, A3 = a * (2 * np.random.random() - 1), a * (2 *  np.random.random() - 1), a * (2 * np.random.random() - 1)
        C1, C2, C3 = 2 *  np.random.random(), 2* np.random.random(), 2* np.random.random()
        X1 = [0.0 for i in range(self.ndims)]
        X2 = [0.0 for i in range(self.ndims)]
        X3 = [0.0 for i in range(self.ndims)]
        pos = [0.0 for i in range(self.ndims)]
        for j in range(self.ndims):
            X1[j] = alphaWolf.pos[j] - A1 * abs(
            C1 * alphaWolf.pos[j] - self.pos[j])
            X2[j] = betaWolf.pos[j] - A2 * abs(
            C2 *  betaWolf.pos[j] - self.pos[j])
            X3[j] = gammaWolf.pos[j] - A3 * abs(
            C3 * gammaWolf.pos[j] - self.pos[j])
            pos[j]+= X1[j] + X2[j] + X3[j]
        for j in range(self.ndims):
            pos[j]/=3.0
        pos = np.clip(pos, self.minval, self.maxval)
        fit = self.fitnessFunction.evaluate(pos)
        if fit < self.fitness:
            self.pos = pos
            self.fitness = fit
        self.reportEvaluation(self)
