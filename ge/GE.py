from __future__ import annotations
from typing import List, Tuple, Callable
from numba import njit
import numpy as np
from .Genome import  Genotype, Phenotype, GenomeMapper


class GE:
    def __init__(self, 
        grammar: str,
        fitnessFunction: Callable[[str], float], 
        populationSize:int=100, 
        numGenerations:int=100, 
        mutationProbability:float=0.1, 
        crossoverProbability:float=0.5, 
        genotypeLength:int=10
    ):
        self.mapper:GenomeMapper = GenomeMapper(grammar)
        self.populationSize:int = populationSize
        self.numGenerations:int = numGenerations
        self.mutationProbability:float = mutationProbability
        self.crossoverProbability:float = crossoverProbability
        self.fitnessFunction:Callable[[str], float] = fitnessFunction
        self.genotypeLength:int = genotypeLength
        self.maxInt = 256

    def evolve(self, expr:str):
        population:List[Individual] = []
        for _ in range(self.populationSize):
            genotype:Genotype = np.random.randint(1,high=self.maxInt+1, size=self.genotypeLength)
            phenotype = self.mapper.genotypeToPhenotype(expr, genotype)
            population.append(
                Individual(
                    genotype, 
                    phenotype,
                    self.fitnessFunction
                )
            )
        for _ in range(self.numGenerations):
            selected = [ self._selection(population, k=int(self.populationSize * self.crossoverProbability)) ]
            childrens = self._mutation(
                self._crossover(selected)
            )
            population = sorted(population,key=lambda g: g.fitness)
            for i in range(self.populationSize - len(childrens)):
                childrens.append(population[i])
            population = childrens
        return sorted(population,key=lambda g: g.fitness)

    def _crossover(self, parents:List[Individual]) -> List[Genotype]:
        childrens = []
        i = 0
        lastIndex = len(parents) - 1
        while(i < lastIndex):
            offspring = _genotypeCrossover(parents[i].genotype, parents[i + 1].genotype)
            childrens.append(offspring[0])
            childrens.append(offspring[1])
            i += 1
        return childrens

    def _mutation(self, childrens:List[Genotype]) -> List[Individual]: 
        mutatedChildrens = []
        for children in childrens:
            for i in range(self.genotypeLength):
                if np.random.uniform(low=0, high = 1) <= self.mutationProbability:
                    children[i] = np.random.randint(1,high=np.int8.max+1)
            mutatedChildrens(
                Individual(
                    children,
                    self.mapper.genotypeToPhenotype(children),
                    self.fitnessFunction
                )
            )
        return mutatedChildrens
    
    #tournament selection procedure
    def _selection(self, population:List[Individual], k:int=2) -> Individual:
        selection_idx = np.random.randint(self.populationSize)
        for ix in np.random.randint(1, self.populationSize, k):
            idx = ix - 1
            if population[idx].fitness < population[selection_idx].fitness:
                selection_idx = idx
        return population[selection_idx]

class Individual: 
    def __init__(self, genotype:Genotype, phenotype:Phenotype, fitFunction:Callable[[str], float]):
        self.genotype:Genotype = genotype
        self.phenotype:Phenotype = phenotype
        self.fitFunction:Callable[[str], float] = fitFunction
        self.fitness:float = fitFunction(phenotype)
  
@njit(nogil=True)
def _genotypeCrossover(genotype1:np.array[np.int8], genotype2:np.array[np.int8], genotypeLength: int) -> Tuple(Genotype):
    if(len(genotype1) != len(genotype2) or len(genotype1) != genotypeLength):
        raise ValueError('Genotype length diferent than expected ', genotypeLength)
    csPoint = np.random.randint(1, genotypeLength)
    return (genotype1[0:csPoint] + genotype2[csPoint:], genotype2[0:csPoint] + genotype1[csPoint:])