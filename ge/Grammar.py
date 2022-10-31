from __future__ import annotations
from typing import Optional
import re, io

class Grammar:
    def __init__(self, fileName: Optional(str) = None):
        self.grammar = {}
        self.grammarRegex = r"^\s*(<{1}[a-zA-Z0-9_-]+>{1})\s*\:{1}\:{1}\={1}\s*((?:.*<{1}[a-zA-Z0-9_-]+>{1}.*)|(?:\s*.+\s*)+\s*\|*\s*)+$"
        if fileName:
            self.loadGrammarFromFile(fileName)
            
    def loadGrammarFromFile(self, fileName: str) -> Grammar:
        file = open(fileName)
        lines = file.readlines()
        self._loadGrammar(lines)
        return self

    def loadGrammar(self, contents: str) -> Grammar:
        buf = io.StringIO(contents)
        lines = buf.readlines()
        self._loadGrammar(lines)
        return self

    def getLeftMostNonTerminal(self, expr: str) -> GrammarExprMatch:
        matches = re.search(r"<{1}[a-zA-Z0-9_-]+>{1}", expr)
        if matches:
            match = matches.group()
            return GrammarExprMatch(match, self.grammar[match])
        return None

    def _loadGrammar(self, lines):
        grammar = {}
        for i in range(len(lines)):
            line = lines[i].strip()
            if line:
                evl =  re.search(self.grammarRegex, line)
                groups = evl.groups() if evl else None
                if groups and len(groups) == 2:
                    grammar[groups[0].strip()] = [str.strip() for str in groups[1].split("|")]
                else:
                    print(lines[i])
                    print(groups)
                    raise ValueError('Invalid rule provided at line ', i+1)
        self.grammar = grammar


class GrammarExprMatch:
    def __init__(self, expr, rules):
        self.rules = rules if rules else []
        self.expr = expr

    def getExpression(self):
        return self.expr

    def replaceFirstOccurrenceOfExpression(self, str, replacement):
        return str.replace(self.expr, replacement)
     
    def getExpressionRules(self):
        return self.rules