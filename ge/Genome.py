from __future__ import annotations
import numpy as np
from .Grammar import *

Genotype = [np.int8]
Phenotype = str

class GenomeMapper:
    def __init__(self, grammarContents: str):
        self.grammar:Grammar = Grammar()
        self.grammar.loadGrammar(grammarContents)

    def genotypeToPhenotype(self, expr:str, genotype:Genotype) -> Phenotype:
        phenotype = expr
        genotypeIndex = 0
        genomeLength = len(genotype)
        while True:
            terminal = self.grammar.getLeftMostNonTerminal(phenotype)
            if terminal:
                rules = terminal.getExpressionRules()
                genotypeIndex = genotypeIndex if genotypeIndex < genomeLength else 0
                replacement = rules[genotype[genotypeIndex] % len(rules)]
                phenotype = terminal.replaceFirstOccurrenceOfExpression(phenotype, replacement)
                genotypeIndex = genotypeIndex + 1
            else:
                break
        return phenotype   

