import sys
from itertools import count
from tree_reader import Tree
from collections import *
import codecs
import simplejson as js

def read(fname,read_trees=True):
    with codecs.open(fname) as fh:
        while True:
            entry = [x.strip() for x in fh.next().strip().split("|||")]
            sid, target, sid2, source, pair = entry
            slen,tlen = map(int,pair.split())
            aligns = []
            for _ in xrange(tlen):
                aa = map(float,fh.next().strip().split())
                aligns.append(aa)
            assert(not fh.next().strip())
            # skip bad trees
            try:
                if read_trees:
                    t = toTree(target.split())
            except AssertionError:
                target = "(TOP (S EMPTY))"
            yield source.split(), target.split(), aligns

def toJson(t):
    t.leftmost()
    t.rightmost()
    if t.isleaf():
        return {'name':t.label, 's':t.s, 'e':t.e, 'sid':t.sid,'eid':t.sid}
    else:
        pref=[{'name':'('}]
        suff=[{'name':')'}]
        children = [toJson(c) for c in t.children]
        return {'name':t.label, 's':t.s, 'e':t.e, 'children':children,'sid':t.sid,'eid':t.eid}

def visit_tree(t):
    yield t
    if t.children:
        for c in t.children:
            for s in visit_tree(c): yield s
        yield t

def toTree(lst):
    toks = [t if t[0] != ")" else ")" for t in lst]
    s = " ".join(toks)
    t = Tree.from_sexpr(s)
    t.annotate_leafs()
    for i,v in enumerate(visit_tree(t)):
        if hasattr(v,'sid'): v.eid=i
        else: v.sid=i
    return t

if __name__ == '__main__':
    # data = list(read("dev_alignments.txt.best.txt"))
    data = list(read("/Users/roeeaharoni/git/research/nmt/models/de_en_stt/dev_alignments.txt.best"))
    sources = [x[0] for x in data]
    targets = [x[1] for x in data]
    trees = [toTree(x) for x in targets]
    alignments = [x[2] for x in data]

    all_alignments = []
    for s,t,a in zip(sources,targets,alignments):
        #assert(len(t)+1==len(a))
        #assert(len(s)+1==len(a[0]))
        all_alignments.append([])
        aligns = all_alignments[-1]
        d2e = defaultdict(list)
        for si,sw in enumerate(s):
            for ti,tw in enumerate(t):
                a_s_t = a[ti][si] # TODO +1?
                typ = "lex"
                if tw[0] == "(": typ = "open"
                if tw[0] == ")": typ = "close"
                aligns.append({'sid':si,'tid':ti,'a':a_s_t,'type':typ})
                if typ == 'lex':
                    d2e[si].append((a_s_t,ti))

        # for the "open" and "close" alignment, add a type pointing to the english aligned word
        for align in aligns[:]:
            if align['type'] == "lex": continue
            ti = align['tid'] # English bracket
            si = align['sid'] # German word
            si2 = max(d2e[si])[1]
            typ = align['type']+"_e"
            aligns.append({'sid':si2,'tid':ti,'a':align['a'],'type':typ})

    max_amount = 1000
    print "TREES = [ %s ]" % ",".join(js.dumps(toJson(x)) for x in trees[:max_amount])
    print "SOURCES = [ %s ]" % ",".join("%s" % x for x in sources[:max_amount])
    print "ALIGNS = %s" % (js.dumps(all_alignments[:max_amount]),)

    # data_bpe = list(read("research/nmt/models/de_en_wmt16/dev_alignments.txt",read_trees=False))
    data_bpe = list(read("/Users/roeeaharoni/git/research/nmt/models/de_en_bpe/dev_alignments.txt.best", read_trees=False))
    sources_bpe = [x[0] for x in data_bpe]
    targets_bpe = [x[1] for x in data_bpe]
    alignments_bpe = [x[2] for x in data_bpe]

    all_alignments_bpe = []
    for s0,t,a in zip(sources_bpe,targets_bpe,alignments_bpe):
        #assert(len(s0) == len(s)), [s,s0]
        assert(len(t)+1==len(a))
        assert(len(s0)+1==len(a[0]))
        all_alignments_bpe.append([])
        aligns = all_alignments_bpe[-1]
        for si,sw in enumerate(s0):
            for ti,tw in enumerate(t):
                a_s_t = a[ti][si] # TODO +1?
                typ = "lex"
                aligns.append({'sid':si,'tid':ti,'a':a_s_t,'type':typ})

    print "SBPES = %s" % (js.dumps(sources_bpe[:max_amount]))
    print "BPES = %s" % (js.dumps(targets_bpe[:max_amount]))
    print "BPE_ALIGN = %s" % (js.dumps(all_alignments_bpe[:max_amount]))
