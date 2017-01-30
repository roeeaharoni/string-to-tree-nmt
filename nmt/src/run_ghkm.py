import os
import codecs
import sys
from itertools import count
from operator import itemgetter

from tree_reader import Tree
from collections import *
import simplejson as js
import numpy


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


def strip_tree(mma, target_labels):
    rows = []
    for i, t in enumerate(target_labels):
        if u'(' not in t.decode('utf-8') and u')' not in t.decode('utf-8') and i < mma.shape[0]:
            try:
                rows.append(mma[i, :])
            except Exception as e:
                print e
    no_tree_matrix = numpy.array(rows)
    no_tree_target_labels = [l for l in target_labels if u'(' not in l.decode('utf-8') and u')' not in l.decode('utf-8')]
    return no_tree_matrix, no_tree_target_labels


def get_ghkm_alignment_file_from_attn_weights(file_path):
    data = list(read("/Users/roeeaharoni/git/research/nmt/models/de_en_stt/dev_alignments.txt.best"))
    sources = [x[0] for x in data]
    targets = [x[1] for x in data]
    trees = [toTree(x) for x in targets]
    alignments = [x[2] for x in data]
    all_alignments = []
    with codecs.open(file_path, 'w', 'utf-8') as output_file:
        for s, t, a in zip(sources, targets, alignments):
            # assert(len(t)+1==len(a))
            # assert(len(s)+1==len(a[0]))

            # remove non-lexical rows
            lexical_alignment_mtx, lexical_targets = strip_tree(numpy.asarray(a), t)
            aligns = ''
            for si, sw in enumerate(s):
                for ti, tw in enumerate(lexical_targets):
                    a_s_t = lexical_alignment_mtx[ti][si]  # TODO +1?
                    if a_s_t > 0.5:
                        aligns += '{}-{} '.format(ti,si)
            aligns = aligns.strip()
            output_file.write(aligns + '\n')



def convert_to_ptb_style(input, output):
    with codecs.open(output, 'w', 'utf-8') as output_file:
        with codecs.open(input, 'r', 'utf-8') as input_file:
            while True:
                line = input_file.readline()
                line = line.replace('(TOP','').replace(')TOP','')
                ptb_toks = []
                for t in line.split():
                    if ')' in t:
                        ptb_toks.append(')')
                    else:
                        if '(' in t:
                            ptb_toks.append(t)
                        else:
                            # wrap all leaves with same pos
                            ptb_toks.append(u'(NNP {})'.format(t))

                ptb_string = ' '.join(ptb_toks)
                output_file.write(ptb_string +'\n')
                if not line:
                    break


def analyze_rules(rules_path):
    parsed = []
    i = 0
    with codecs.open(rules_path, 'r', 'utf-8') as rules_file:
        while i < 46220:
            i+=1
            print i
            line = rules_file.readline()
            parts = line.split("|||")
            if len(parts) != 2:
                continue
            scores = parts[1].strip().split(' ')
            parsed.append((parts[0], float(scores[0]), float(scores[1])))
            if i > 46220:
                break

    sort = parsed.sort(key=itemgetter(1),reverse=False)
    for p in parsed:
        print p



def main():
    input_trees_path = '/Users/roeeaharoni/git/research/nmt/models/de_en_stt/newstest2015-deen-src.tok.true.de.bpe.output.trees.dev.best'
    output_trees_path = '/Users/roeeaharoni/git/research/nmt/models/de_en_stt/ghkm/de_en_stt.ptb'
    # convert_to_ptb_style(input_trees_path, output_trees_path)
    # get_ghkm_alignment_file_from_attn_weights('/Users/roeeaharoni/git/research/nmt/models/de_en_stt/ghkm/de_en_stt.a')

    ghkm_path = '~/git/galley-ghkm/'
    files_path = '~/git/research/nmt/models/de_en_stt/ghkm/de_en_stt'
    rules_path = '/Users/roeeaharoni/git/research/nmt/models/de_en_stt/ghkm/rules.txt'
    os.system(ghkm_path + 'extract.sh {} 1g false > {}'.format(files_path, rules_path))

    # cool example: VP(x0:NNP x1:NP) -> x1 x0 ||| 0.011629093 0.13380282
    analyze_rules(rules_path)

    return

if __name__ == '__main__':
    main()