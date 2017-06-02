import re
import codecs
import time
from collections import Counter
import random

def _tokenize_sexpr(s):
    tokker = re.compile(r" +|[()]|[^ ()]+")
    toks = [t for t in [match.group(0) for match in
tokker.finditer(s)] if t[0] != " "]
    return toks

def _within_bracket(toks):
    label = next(toks)
    children = []
    for tok in toks:
        if tok == "(":
            children.append(_within_bracket(toks))
        elif tok == ")":
            return Tree(label, children)
        else: children.append(Tree(tok, None))
    assert(False),list(toks)

class Tree(object):
    def __init__(self, label, children=None):
        self.label = label
        self.children = children
        self.s = None
        self.e = None

    @staticmethod
    def from_sexpr(string):
        toks = iter(_tokenize_sexpr(string))
        assert next(toks) == "("
        return _within_bracket(toks)

    def __str__(self):
        if self.children is None: return self.label
        return "[%s %s]" % (self.label, " ".join([str(c) for c in
self.children]))

    def isleaf(self): return self.children==None
    def leftmost(self):
        if self.s is None:
            self.s = self.children[0].leftmost()
        return self.s

    def rightmost(self):
        if self.e is None:
            self.e = self.children[-1].rightmost()
        return self.e

    def leaves_iter(self):
        if self.isleaf():
            yield self
        else:
            for c in self.children:
                for l in c.leaves_iter(): yield l

    def leaves(self): return list(self.leaves_iter())

    def nonterms_iter(self):
        if not self.isleaf():
            yield self
            for c in self.children:
                for n in c.nonterms_iter(): yield n

    def nonterms(self): return list(self.nonterms_iter())

    def annotate_leafs(self):
        for i,lf in enumerate(self.leaves(),1):
            lf.s = i
            lf.e = i


    def asJSON(self):
        if self.isleaf(): return '{name: "%s"}' % self.label
        children = "[ %s ]" % ",".join(c.asJSON() for c in self.children)
        return '{name: "%s", children: %s }' % (self.label, children)


def read_dataset(filename):
    return [Tree.from_sexpr(line.strip()) for line in codecs.open(filename,"r")]



