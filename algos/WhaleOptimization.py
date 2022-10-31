from __future__ import annotations
from typing import Optional
import copy
import numpy as np
from utils import FitnessFunction
from .SiAlgo import *

class WhaleOptimization(SiAlgo):
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
        self.spread = self.options["spread"] if "spread" in self.options else 2
        self.spreadStep = self.spread/numGenerations
        self.spiral = np.clip(self.options["spiral"] if "spiral" in self.options else .5, 0, 1)

    def search(self) -> SiSearchResult:
        self.resetEvaluations()
        reportEval = lambda p: self.appendAndAssertEvaluation(p.fitness) 
        wales:List[Whale] = sorted([Whale(self.fitnessFunction, reportEval, self.spiral, i) for i in range(self.populationSize)], key = lambda w: w.fitness)
        best:Whale = wales[0]
        generations:List[float] = []
        try:
            for _ in range(self.numGenerations):
                newWales:List[Whale] = np.copy(wales)
                for w in newWales:
                    p:float = np.random.uniform(0.0, 1.0)
                    if p > .5: 
                        r = np.random.uniform(0.0, 1.0, size=w.ndims)   
                        A = (2.0*np.multiply(self.spread, r))- self.spread
                        normA = np.linalg.norm(A)     
                        if(normA < 1):
                            w.encircle(best, A)     
                        else:
                            w.search(wales[randIndex(self.populationSize, excludeIndex=w.index)], A)
                    else:
                        w.attack(best)
                wales = sorted(newWales, key = lambda temp: temp.fitness)
                best:Whale = wales[0]
                self.spread -=  self.spreadStep
                generations.append(best.fitness)
        except MaxEvaluationReached:
            None   
        return SiSearchResult(
            best=best,
            population=wales, 
            fitnessByGeneration=generations,
            fitnessByEvaluation=self.evaluations
        )


class Whale(SiAgent):
    def __init__(self, fitnessFunction:FitnessFunction, reportEvaluation:Callable, spiral: float, index: int):
        super().__init__(fitnessFunction)
        bounds = fitnessFunction.bounds()
        self.minval:float = bounds.min
        self.maxval:float = bounds.max
        self.ndims:int = fitnessFunction.dimensionsLength()
        self.pos = np.random.uniform(low=self.minval, high = self.maxval, size = self.ndims)
        self.best:List[float] = self.pos
        self.fitness = fitnessFunction.evaluate(self.pos)
        self.reportEvaluation: Callable = reportEvaluation
        self.index:int = index
        self.spiral:float = spiral
        reportEvaluation(self)
     
    def search(self, randomWhale:Whale, A:List[float]):
        C = 2.0*np.random.uniform(0.0, 1.0, size=self.ndims)
        D = np.linalg.norm(np.multiply(C, randomWhale.pos) - self.pos)    
        pos = randomWhale.pos - np.multiply(A, D)
        self.update(pos)

    def attack(self, best:Whale):
        D = np.linalg.norm(best.pos - self.pos)
        L = np.random.uniform(-1.0, 1.0, size=self.ndims)
        pos = np.multiply(np.multiply(D,np.exp(self.spiral*L)), np.cos(2.0*np.pi*L))+best.pos
        self.update(pos)

    def encircle(self, best:Whale, A:List[float]):
        C =  2.0*np.random.uniform(0.0, 1.0, size=self.ndims)
        D = np.linalg.norm(np.multiply(C, best.pos)  - self.pos)
        pos = best.pos - np.multiply(A, D)
        self.update(pos)

    def update(self, pos:List[float]):
        self.pos = np.clip(pos, self.minval, self.maxval) 
        self.fitness = self.fitnessFunction.evaluate(self.pos)
        self.reportEvaluation(self)

def randIndex(length, excludeIndex = None):
    index = np.random.randint(low=0, high=length)  
    if index == excludeIndex:
        return randIndex(length, excludeIndex)
    return index
