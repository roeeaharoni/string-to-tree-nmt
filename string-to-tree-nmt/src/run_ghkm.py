import os
import codecs
import sys
from itertools import count
from operator import itemgetter

from tree_reader import Tree
from collections import *
import simplejson as js
import numpy

import re


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

            for ti, tw in enumerate(lexical_targets):
                max_align = numpy.argmax(lexical_alignment_mtx[ti])

                if max_align != len(lexical_alignment_mtx[ti]) - 1:
                    aligns += '{}-{} '.format(ti,max_align)

                for si, sw in enumerate(s[:-1]):
                    a_s_t = lexical_alignment_mtx[ti][si]
                    if a_s_t > 0.3 and si != max_align and max_align != len(lexical_alignment_mtx[ti]) - 1:
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
                            ptb_toks.append(u'(TER {})'.format(t))

                ptb_string = ' '.join(ptb_toks)
                output_file.write(ptb_string +'\n')
                if not line:
                    break


def analyze_rules(rules_path):
    sorted_rules_path = rules_path + '.sorted.txt'
    sorted_reordering_rules_path = rules_path + '.reordering.sorted.txt'
    parsed = []
    i = 0
    with codecs.open(rules_path, 'r', 'utf-8') as rules_file:
        while True:
            line = rules_file.readline()
            i+=1
            if i==1:
                continue
            print i
            if not line:
                break
            else:
                parts = line.split("|||")
                scores = parts[1].strip().split(' ')
                parsed.append((parts[0], float(scores[0]), float(scores[1]), float(scores[2])))

    sort = parsed.sort(key=itemgetter(3),reverse=True)
    with codecs.open(sorted_rules_path, 'w', 'utf-8') as sorted_rules_file:
        with codecs.open(sorted_reordering_rules_path, 'w', 'utf-8') as sorted_reordering_rules_file:
            reoroderd = 0
            for p in parsed:
                r = re.compile(u"(x[0-9])")
                m = re.findall(r, p[0].split('->')[1])

                if len(m)>0:
                    i = 0

                    for s in m:
                        index = int(s.replace('x',''))
                        if index!=i:
                            # print u'{}\t{}\t{}\t{}\n'.format(p[0],p[1],p[2],p[3])
                            reoroderd +=1
                            sorted_reordering_rules_file.write(u'{}\t{}\t{}\t{}\n'.format(p[0],p[1],p[2],p[3]))
                            break
                        else:
                            i += 1
                sorted_rules_file.write(u'{}\t{}\t{}\t{}\n'.format(p[0], p[1], p[2], p[3]))
        print '{} reordered out of {}'.format(reoroderd, len(parsed))


def ghkm2sentid(input, output):
    rule2sents = {}
    with codecs.open(input, 'r', 'utf-8') as file:
        while True:
            line = file.readline()
            if not line:
                break
            s = line.split('\t')
            id = s[0]
            rule = s[1].strip()
            if rule in rule2sents:
                rule2sents[rule].append(id)
            else:
                rule2sents[rule] = [id]

    tuples = []
    for r in rule2sents:
        tuples.append((r, rule2sents[r], len(rule2sents[r])))

    tuples.sort(key=itemgetter(2), reverse=True)

    with codecs.open(output, 'w', 'utf-8') as outfile:
        for t in tuples:
            outfile.write(u"{}\t{}\t{}\n".format(t[0],t[2],t[1]))
            print str(t) + '\n'


def filter_where_more_reordering(indices_file, alignments_file, source_file, trees_file):
    with codecs.open(indices_file, 'r', 'utf-8') as indices:
        lines = indices.readlines()
        line_numbers = [int(l.strip()) for l in lines]

    with codecs.open(alignments_file, 'r', 'utf-8') as alignments:
        with codecs.open(source_file, 'r', 'utf-8') as source:
            with codecs.open(trees_file, 'r', 'utf-8') as trees:
                a = alignments.readlines()
                s = source.readlines()
                t = trees.readlines()
                with codecs.open(alignments_file + '.filtered', 'w', 'utf-8') as alignments_f:
                    with codecs.open(source_file + '.filtered', 'w', 'utf-8') as source_f:
                        with codecs.open(trees_file + '.filtered', 'w', 'utf-8') as trees_f:
                            for number in line_numbers:
                                alignments_f.write(a[number])
                                source_f.write(s[number])
                                trees_f.write(t[number])






def main():

    filter_where_more_reordering('/Users/roeeaharoni/git/research/nmt/models/de_en_stt/higher_reordering_ids.txt',
                                 '/Users/roeeaharoni/git/research/nmt/models/de_en_stt/ghkm/de_en_stt.a',
                                 '/Users/roeeaharoni/git/research/nmt/models/de_en_stt/ghkm/de_en_stt.f',
                                 '/Users/roeeaharoni/git/research/nmt/models/de_en_stt/ghkm/de_en_stt.ptb')
    return

    ghkm2sentid("/Users/roeeaharoni/git/research/nmt/models/de_en_stt/ghkm/ghkm2sent_max_att.txt",
                "/Users/roeeaharoni/git/research/nmt/models/de_en_stt/ghkm/ghkm2sent_max_att_ids.txt")
    return

    input_trees_path = '/Users/roeeaharoni/git/research/nmt/models/de_en_stt/newstest2015-deen-src.tok.true.de.bpe.output.trees.dev.best'
    output_trees_path = '/Users/roeeaharoni/git/research/nmt/models/de_en_stt/ghkm/de_en_stt.ptb'

    # convert_to_ptb_style(input_trees_path, output_trees_path)
    get_ghkm_alignment_file_from_attn_weights('/Users/roeeaharoni/git/research/nmt/models/de_en_stt/ghkm/de_en_stt.a')

    ghkm_path = '~/git/galley-ghkm/'
    files_path = '~/git/research/nmt/models/de_en_stt/ghkm/de_en_stt'
    rules_path = '/Users/roeeaharoni/git/research/nmt/models/de_en_stt/ghkm/rules_max_att.txt'
    os.system(ghkm_path + 'extract.sh {} 1g false > {}'.format(files_path, rules_path))

    # cool example: VP(x0:NNP x1:NP) -> x1 x0 ||| 0.011629093 0.13380282
    analyze_rules(rules_path)



if __name__ == '__main__':
    main()