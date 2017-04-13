import random

import numpy
import sys
import os
import json
import argparse
import math
import codecs
from collections import defaultdict
from collections import Counter
from operator import itemgetter

import matplotlib.pyplot as plt
from matplotlib import gridspec

from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data

import pydot

import plotly

import yoav_trees



# input:
#  alignment matrix - numpy array
#  shape (target tokens + eos, number of hidden source states = source tokens +eos)
# one line correpsonds to one decoding step producing one target token
# each line has the attention model weights corresponding to that decoding step
# each float on a line is the attention model weight for a corresponding source state.
# plot: a heat map of the alignment matrix
# x axis are the source tokens (alignment is to source hidden state that roughly corresponds to a source token)
# y axis are the target tokens

# http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
def plot_heat_map(attn_mtx, target_labels, source_labels, attn_mtx2=None, target_labels2=None, source_labels2=None,
                  id=None, file=None, image=None):
    plt.ioff()

    stripped_attn_mtx, stripped_target_labels = strip_tree(attn_mtx, target_labels)

    if attn_mtx2 == None:
        fig, ax = plt.subplots()
    else:
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(20, 10))
        # ax = plt.subplot(221)
        ax = fig.add_subplot(gs[:, 0])

        score = get_diagonal_subsequent_reordering_score(stripped_attn_mtx, source_labels, stripped_target_labels)
        ax.text(float(len(source_labels)) / 2, len(stripped_target_labels) + 1, 'bpe2tree:{}'.format(score))

    heatmap = ax.pcolor(stripped_attn_mtx, cmap=plt.cm.Blues)

    # put the major ticks at the middle of each cell
    ax.set_xticks(numpy.arange(stripped_attn_mtx.shape[1]) + 0.5, minor=False)
    ax.set_yticks(numpy.arange(stripped_attn_mtx.shape[0]) + 0.5, minor=False)

    # without this I get some extra columns rows
    # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
    ax.set_xlim(0, int(stripped_attn_mtx.shape[1]))
    ax.set_ylim(0, int(stripped_attn_mtx.shape[0]))

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # source words -> column labels
    ax.set_xticklabels(source_labels, minor=False)
    # target words -> row labels
    ax.set_yticklabels(stripped_target_labels, minor=False)

    plt.xticks(rotation=45)

    if attn_mtx2 is not None:
        score = get_diagonal_subsequent_reordering_score(attn_mtx2, source_labels2, target_labels2)
        # ax2 = plt.subplot(222)
        ax2 = fig.add_subplot(gs[:, 1])
        ax2.text(float(len(source_labels2)) / 2, len(target_labels2) + 1, 'bpe2bpe:{}'.format(score))
        heatmap = ax2.pcolor(attn_mtx2, cmap=plt.cm.Blues)

        # put the major ticks at the middle of each cell
        ax2.set_xticks(numpy.arange(attn_mtx2.shape[1]) + 0.5, minor=False)
        ax2.set_yticks(numpy.arange(attn_mtx2.shape[0]) + 0.5, minor=False)

        # without this I get some extra columns rows
        # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
        ax2.set_xlim(0, int(attn_mtx2.shape[1]))
        ax2.set_ylim(0, int(attn_mtx2.shape[0]))

        # want a more natural, table-like display
        ax2.invert_yaxis()
        ax2.xaxis.tick_top()

        # source words -> column labels
        ax2.set_xticklabels(source_labels2, minor=False)
        # target words -> row labels
        ax2.set_yticklabels(target_labels2, minor=False)

        plt.xticks(rotation=45)

    if file != None:
        plt.savefig(file)
        plt.close()

        # plt.show()
    else:
        plt.show()


    # show first figure again separately
    if attn_mtx2 is not None or image != None:
        gs = gridspec.GridSpec(2, 2)
        if image != None:
            # fig = plt.figure(figsize=(20, 10))
            # ax3 = fig.add_subplot(gs[:, :])
            fig, ax3 = plt.subplots()

            fn = get_sample_data(image, asfileobj=False)
            arr_img = plt.imread(fn, format='png')

            imagebox = OffsetImage(arr_img, zoom=0.6)
            imagebox.image.axes = ax

            xy = (0.5, 0.5)
            ab = AnnotationBbox(imagebox, xy,
                                xybox=(0, 0),
                                xycoords='data',
                                boxcoords="offset points",
                                # pad=0.5
                                )

            ax3.add_artist(ab)
            plt.close()

        else:
            # show full tree matrix
            fig = plt.figure(figsize=(20, 10))
            score = get_diagonal_subsequent_reordering_score(attn_mtx, source_labels, target_labels)
            # ax3 = plt.subplot(224)
            ax3 = fig.add_subplot(gs[:, :])
            ax3.text(float(len(source_labels)) / 2, len(target_labels) + 1, 'bpe2bpe:{}'.format(score))
            heatmap = ax3.pcolor(attn_mtx, cmap=plt.cm.Blues)

            # put the major ticks at the middle of each cell
            ax3.set_xticks(numpy.arange(attn_mtx.shape[1]) + 0.5, minor=False)
            ax3.set_yticks(numpy.arange(attn_mtx.shape[0]) + 0.5, minor=False)

            # without this I get some extra columns rows
            # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
            ax3.set_xlim(0, int(attn_mtx.shape[1]))
            ax3.set_ylim(0, int(attn_mtx.shape[0]))

            # want a more natural, table-like display
            ax3.invert_yaxis()
            ax3.xaxis.tick_top()

            # source words -> column labels
            ax3.set_xticklabels(source_labels, minor=False)
            # target words -> row labels
            ax3.set_yticklabels(target_labels, minor=False)

            plt.xticks(rotation=45)

            if file != None:
                plt.savefig(file+'.tree.png')
                plt.close()

            plt.show()
            plt.close()

        # plt.tight_layout()
        # if id and id > 1213:
        #     print 'saved {} to file'.format(id)


    return


# column labels -> target words
# row labels -> source words
def read_alignment_matrix(f):
    header = f.readline().strip().split('|||')
    if header[0] == '':
        return None, None, None, None
    sid = int(header[0].strip())
    # number of tokens in source and translation +1 for eos
    src_count, trg_count = map(int, header[-1].split())
    # source words
    source_labels = header[3].decode('UTF-8').split()
    source_labels.append('</s>')
    # target words
    target_labels = header[1].decode('UTF-8').split()
    target_labels.append('</s>')

    mm = []
    for r in range(trg_count):
        alignment = map(float, f.readline().strip().split())
        mm.append(alignment)
    mma = numpy.array(mm)
    return sid, mma, target_labels, source_labels


def get_distortion_step_sizes(file1, file2):
    count=0
    file1_steps = []
    file2_steps = []
    while (file1):
        count += 1
        print count
        sid, mma, target_labels, source_labels = read_alignment_matrix(file1)
        if not target_labels:
            break

        # remove rows of tree symbols
        no_tree_matrix, no_tree_target_labels = strip_tree(mma, target_labels)

        file1_steps += get_binned_step_sizes(no_tree_matrix, source_labels, no_tree_target_labels)

        # compare to WMT model
        if file2:
            sid2, mma2, target_labels2, source_labels2 = read_alignment_matrix(file2)
            if target_labels2 is None:
                print 'no target labels!'
            else:
                file2_steps += get_binned_step_sizes(mma2, source_labels2, target_labels2)

        # read next
        line = file1.readline()
        if not line:
            break
        line2 = file2.readline()
        if not line2:
            break
    cnt1 = Counter(file1_steps)
    cnt2 = Counter(file2_steps)
    cnt1 = dict(cnt1)
    cnt2 = dict(cnt2)
    for i in xrange(100):
        if float(i) in cnt1:
            print i, ' ', cnt1[float(i)]
        else:
            print i, ' ', 0

    print "========================"
    print "========================"
    print "========================"
    print "========================"
    for i in xrange(100):
        if float(i) in cnt2:
            print i, ' ', cnt2[float(i)]
        else:
            print i, ' ', 0
    return cnt1, cnt2


def strip_tree(mma, target_labels):
    rows = []
    for i, t in enumerate(target_labels):
        if u'(' not in t and u')' not in t and i < mma.shape[0]:
            try:
                rows.append(mma[i, :])
            except Exception as e:
                print e
    no_tree_matrix = numpy.array(rows)
    no_tree_target_labels = [l for l in target_labels if u'(' not in l and u')' not in l]
    return no_tree_matrix, no_tree_target_labels


def get_binned_step_sizes(attn_matrix, input_lables, output_labels):
    step_sizes = []
    x, y = attn_matrix.shape
    prev_highest_index = 0
    for diag_y_index, y in enumerate(output_labels):
        # measure distance between every two consequent words
        if diag_y_index < attn_matrix.shape[0]:
            row = attn_matrix[diag_y_index, :]
            highest_index = numpy.argmax(row)
            dist = math.fabs(highest_index - prev_highest_index)
            step_sizes.append(dist)
            prev_highest_index = highest_index

    return step_sizes


def get_diagonal_distance_reordering_score(attn_matrix, input_lables, output_labels):
    x, y = attn_matrix.shape
    in_len = x
    out_len = y
    sum_dists = 0
    for diag_y_index, y in enumerate(output_labels):
        diag_x_index = math.floor(float(in_len - 1) / (out_len - 1) * diag_y_index)
        # measure distance of highest scoring weight from diag_x_index
        if diag_y_index < attn_matrix.shape[0]:
            row = attn_matrix[diag_y_index, :]
            highest_index = numpy.argmax(row)
            dist = math.fabs(highest_index - diag_x_index)
            sum_dists += dist
        else:
            # count one element less
            out_len -= 1

    normalized = sum_dists / out_len
    return normalized

# average distance between two subsequent max attn weights
def get_diagonal_subsequent_reordering_score(attn_matrix, input_lables, output_labels):
    x, y = attn_matrix.shape
    in_len = x
    out_len = y
    sum_dists = 0
    prev_highest_index = 0
    for diag_y_index, y in enumerate(output_labels):
        # measure distance between every two consequent words
        if diag_y_index < attn_matrix.shape[0]:
            row = attn_matrix[diag_y_index, :]
            highest_index = numpy.argmax(row)
            dist = math.fabs(highest_index - prev_highest_index)
            prev_highest_index = highest_index
            sum_dists += dist
        else:
            # count one element less
            out_len -= 1

    normalized = sum_dists / out_len
    return normalized


# iterate through the alignment matrices from one/two models
def inspect_alignment_matrices(file1, file2=None):
    count = 0
    syntax_based_higher = 0
    bpe_based_higher = 0
    high_bpe = defaultdict(int)
    high_syntax = defaultdict(int)
    while (file1):
        count += 1
        # print count
        sid, mma, target_labels, source_labels = read_alignment_matrix(file1)
        if not target_labels:
            break


        no_tree_matrix, no_tree_target_labels = strip_tree(mma, target_labels)

        tree_score = get_diagonal_subsequent_reordering_score(no_tree_matrix, source_labels, no_tree_target_labels)

        # compare to WMT model
        if file2:
            sid2, mma2, target_labels2, source_labels2 = read_alignment_matrix(file2)
            if target_labels2 is None:
                print 'no target labels!'

            bpe_score = get_diagonal_subsequent_reordering_score(mma2, source_labels2, target_labels2)

        # if tree_score > 4:
        to_show = [602, 261, 39, 1227, 1146, 614, 943, 1135, 415, 865]
        if (count - 1) in to_show:
            plot_heat_map(mma, target_labels, source_labels, mma2, target_labels2, source_labels2, count)

        if bpe_score < tree_score:
            print count
            syntax_based_higher += 1
        else:
            bpe_based_higher += 1
            # (target_len, source_len) = no_tree_matrix.shape

        for k in xrange(30):
            # if tree_score == 0:
            #     print 'neg'
            if bpe_score >= k and bpe_score < k+1:
                high_bpe[k] += 1
            if tree_score >= k and tree_score < k+1:
                high_syntax[k] += 1

        # image_path_prefix = '/Users/roeeaharoni/git/research/nmt/src/plots/'
        # file_path = image_path_prefix + str(count) + '.png'
        # if count == 603:
        #     plot_heat_map(mma, target_labels, source_labels, mma2, target_labels2, source_labels2, file=file_path)


        # if comp and bpe_score < tree_score:
        # plot both alignments on same figure for comparison
        # plot_heat_map(no_tree_matrix, no_tree_target_labels, source_labels, mma2, target_labels2, source_labels2)
        # else:
        #     plot_heat_map(no_tree_matrix, no_tree_target_labels, source_labels)

        # plots full tree alignment
        # if mma is None:
        #   return
        # if sid >n:
        #   return
        # (target_len,source_len) = mma.shape
        # if source_len < 20:
        #     plot_head_map(mma, target_labels, source_labels)

        # empty line separating the matrices
        line = file1.readline()
        if not line:
            break
        line2 = file2.readline()
        if not line2:
            break

    print 'syntax had more reordering in: ', syntax_based_higher
    print 'bpe had more reordering in: ', bpe_based_higher

    print 'bpe high'
    for k in xrange(30):
        print k, '\t', high_bpe[k]
    print 'tree high'
    for k in xrange(30):
        print k, '\t', high_syntax[k]
        # print 'bpe high {}'.format(high_bpe)
        # print 'syntax high {}'.format(high_syntax)

    return


def compute_sent_level_bleu_scores(ref, hyp):
    scores = []
    ref_tmp = ref + '.tmp'
    hyp_tmp = hyp + '.tmp'
    res_tmp = 'bleu.tmp'

    bleu_command_format = '~/git/mosesdecoder/scripts/generic/multi-bleu.perl -lc {0} < {1} > {2}'

    # open files
    with codecs.open(ref, 'r', 'utf-8') as ref_file:
        with codecs.open(hyp, 'r', 'utf-8') as hyp_file:
            ref_line = ref_file.readline()
            hyp_line = hyp_file.readline()
            while ref_line:
                # write each row to separate file
                with codecs.open(ref_tmp, 'w', 'utf-8') as ref_row:
                    with codecs.open(hyp_tmp, 'w', 'utf-8') as hyp_row:
                        ref_row.write(ref_line)
                        hyp_row.write(hyp_line)

                # compute bleu for row
                os.system(bleu_command_format.format(ref_tmp, hyp_tmp, res_tmp))

                # save the score
                with codecs.open(res_tmp, 'r', 'utf-8') as res_file:
                    raw = res_file.readline().strip()
                    scores.append((ref_line, hyp_line, raw))

                ref_line = ref_file.readline()
                hyp_line = hyp_file.readline()

    return scores


def compare_sentence_level_bleu():

    scores_tree = compute_sent_level_bleu_scores(
        '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-ref.en',
        '/Users/roeeaharoni/git/research/nmt/models/de_en_stt/newstest2015-deen-src.tok.true.de.bpe.output.sents.dev.postprocessed.best')

    scores_bpe = compute_sent_level_bleu_scores(
        '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-ref.en',
        '/Users/roeeaharoni/git/research/nmt/models/de_en_bpe/newstest2015-deen-src.tok.true.de.bpe.output.dev.postprocessed.best')

    # scores_bpe = compute_sent_level_bleu_scores(
    #     '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-ref.en',
    #     '/Users/roeeaharoni/git/research/nmt/models/de_en_wmt16/newstest2015-deen-src.de.output.en')

    source_sentences = codecs.open('/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-src.de',
                                   'r', 'utf-8').readlines()

    output_trees = codecs.open('/Users/roeeaharoni/git/research/nmt/models/de_en_stt/newstest2015-deen-src.tok.true.de.bpe.output.trees.dev.best',
                                   'r', 'utf-8').readlines()

    diffs = []
    for i in xrange(len(scores_tree)):
        unigram_score_tree = scores_tree[i][2].split(',')[1].split('/')[0].strip()
        bleu_score_tree = scores_tree[i][2].split(',')[0].replace('BLEU = ', '')

        unigram_score_bpe = scores_bpe[i][2].split(',')[1].split('/')[0].strip()
        bleu_score_bpe = scores_bpe[i][2].split(',')[0].replace('BLEU = ', '')

        acc_diff = float(bleu_score_tree) - float(bleu_score_bpe)
        diffs.append({'raw_tree_bleu': scores_tree[i][2],
                      'raw_bpe_bleu': scores_bpe[i][2],
                      'diff': acc_diff,
                      'src': source_sentences[i],
                      'ref': scores_tree[i][0],
                      'stripped_tree': scores_tree[i][1],
                      'tree': output_trees[i],
                      'tree_score': scores_tree[i][2],
                      'bpe': scores_bpe[i][1],
                      'bpe_score': scores_bpe[i][2],
                      'id':i})


    # large positive diffs = bpe is better
    out_format = u'id: {}\n\ndiff: {}\n\nsrc:\n{}\nref:\n{}\nstripped tree ({}):\n{}\ntree:\n{}\nbpe ({}):\n{} \n\n'

    comparison_path = '/Users/roeeaharoni/git/research/nmt/models/de_en_stt/bleu_comparison.txt'
    diffs.sort(key=lambda x: x['diff'],reverse=True)
    with codecs.open(comparison_path, 'w', 'utf-8') as comparison_file:
        for i, d in enumerate(diffs):
            output = out_format.format(d['id'],
                                       d['diff'],
                                        d['src'],
                                        d['ref'],
                                        d['raw_tree_bleu'],
                                        d['stripped_tree'],
                                        d['tree'],
                                        d['raw_bpe_bleu'],
                                        d['bpe'])
            print output
            comparison_file.write(output)


    # top = diffs[:1000]
    # bottom = diffs[-1000:]
    #
    # for i, t in enumerate(top):
    #     output = out_format.format(i,
    #                                t['diff'],
    #                                t['src'],
    #                                t['ref'],
    #                                t['raw_tree_bleu'],
    #                                t['stripped_tree'],
    #                                t['tree'],
    #                                t['raw_bpe_bleu'],
    #                                t['bpe'])
    #     print output
    # # large negative diffs = tree is better
    # for i, t in enumerate(bottom):
    #     output = out_format.format(i,
    #                                t['diff'],
    #                                t['src'],
    #                                t['ref'],
    #                                t['raw_tree_bleu'],
    #                                t['stripped_tree'],
    #                                t['tree'],
    #                                t['raw_bpe_bleu'],
    #                                t['bpe'])
    #     print output


def rec_get_subtree_indices(tree, initial_index):
    leaves = tree.leaves()
    index = initial_index
    child_spans = []

    if tree.children is None:
        return []
        # return [(initial_index, initial_index)]

    if tree.children is not None:
        for c in tree.children:
            child_spans += rec_get_subtree_indices(c, index)
            if c.leaves() is not None:
                index += len(c.leaves())
            else:
                index += 1

    return [(tree, initial_index, initial_index + len(leaves) - 1)] + child_spans



def align_subtrees_to_spans(tree_target_labels):
    parseable = []
    for t in tree_target_labels:
        if ')' in t:
            parseable.append(')')
        else:
            parseable.append(t)
    tree_string = ' '.join(parseable)
    try:
        tree = yoav_trees.Tree('TOP').from_sexpr(tree_string)
        subtree_indices = rec_get_subtree_indices(tree, 0)
    except Exception as e:
        print 'invalid tree'
        return []



    # for st in subtree_indices:
    #     print u'({},{}) {}'.format(st[1], st[2], st[0])

    # print tree

    # leaves = tree.leaves()
    # for i in xrange(len(leaves)):
    #     print i, ' ', leaves[i]

    return subtree_indices


def get_subtrees_with_reordering(file1):

    count = 0
    subtrees_with_reordering_scores = []
    # go through tree attn matrices
    while (file1):
        count += 1
        print count
        sid, mma, linearized_tree, source_labels = read_alignment_matrix(file1)
        if not linearized_tree:
            break

        no_tree_matrix, no_tree_target_labels = strip_tree(mma, linearized_tree)

        subtrees = align_subtrees_to_spans(linearized_tree)

        tree_distortion_scores = []
        for sub in subtrees:
            subtree = sub[0]
            start_row_index = sub[1]
            end_row_index = sub[2]
            if start_row_index < end_row_index:
                subtree_matrix = no_tree_matrix[start_row_index:end_row_index+1,:]
                subtree_target_labels = no_tree_target_labels[start_row_index:end_row_index+1]
                subtree_distortion_score = get_diagonal_subsequent_reordering_score(subtree_matrix,
                                                                                    source_labels,
                                                                                    subtree_target_labels)
                tree_distortion_scores.append((subtree_distortion_score, subtree, start_row_index, end_row_index))
                plot_tree_with_alignments(subtree, subtree_matrix, source_labels, subtree_target_labels)
                # plot_heat_map(subtree_matrix, subtree_target_labels, source_labels)
            # else:
            #     print 'leaf'

            subtrees_with_reordering_scores += tree_distortion_scores

        line = file1.readline()
        if not line:
            break

    return subtrees_with_reordering_scores
    # in each tree

    #   for each span (subtree)

    #       measure reordering amount
    #       return span, reordering amount

    # create bins of reordering amount by span type 1-level - VP/NP/PP...

    # create bins of reordering amount by span type 2-level - X -> Y Z

    # create bins of reordering amount by tree depth


    # what do we want to see (collins 2005):

        # 1 Verb Initial - lots of reordering in VPs - verb (head) will move to the beginning of VP

        # 2 Verb 2nd - reordering in general clauses (S's) - head will move to be after the complementizer ('that ...' etc.)

        # 3 Move Subject - reoredering in S's - move subject to precede the head (verb)

        # 4 Particles - reordering in VP's (or any clause) - move verb particle (on, off, up) to be before the verb

        # 5 Infinitives - move infinite verbs *directly* after finite verbs - can't submit, will look into

        # 6 Negation - could hand it in not -> could not hand it in

    return


def populate_pydot_graph(tree, graph, parentnode):
    # node = pydot.Node(tree.label, style="filled", fillcolor="white", color="white")
    # graph.add_node(node)

    if tree.children != None:
        for child in tree.children:
            r = random.randint(0, 999999)
            label = child.label
            label = label.replace(',','COMMA')
            if child.children == None:
                fcolor = "navy"
            else:
                fcolor= "black"
            child_node = pydot.Node(label + str(r),label=label, style="filled", fillcolor="white",
                                    color="white",
                                    fontcolor=fcolor)
            graph.add_node(child_node)
            graph.add_edge(pydot.Edge(parentnode, child_node))
            graph = populate_pydot_graph(child, graph, child_node)

    return graph


def plot_tree_with_alignments(tree, alignments_mtx, input_labels, output_labels):
    graph = pydot.Dot(graph_type='graph')
    rootnode = pydot.Node(tree.label,label=tree.label, style="filled", fillcolor="white", color="white")
    graph = populate_pydot_graph(tree, graph, rootnode)

    try:
        graph.write_png('tree.png')
    except Exception as e:
        print e

    plot_heat_map(alignments_mtx, output_labels, input_labels, file='alignments.png', image='/Users/roeeaharoni/git/research/nmt/src/tree.png')
    return


def distortion_over_time():
    stt_model_files = [
        'de_en_stt_raw_model.iter1230000.npz',
        'de_en_stt_raw_model.iter1200000.npz',
        'de_en_stt_raw_model.iter1170000.npz',
        'de_en_stt_raw_model.iter1140000.npz',
        'de_en_stt_raw_model.iter1110000.npz',
        'de_en_stt_raw_model.iter1080000.npz',
        'de_en_stt_raw_model.iter1050000.npz',
        'de_en_stt_raw_model.iter1020000.npz',
        'de_en_stt_raw_model.iter990000.npz',
        'de_en_stt_raw_model.iter960000.npz',
        'de_en_stt_raw_model.iter930000.npz',
        'de_en_stt_raw_model.iter900000.npz',
        'de_en_stt_raw_model.iter870000.npz',
        'de_en_stt_raw_model.iter840000.npz',
        'de_en_stt_raw_model.iter810000.npz',
        'de_en_stt_raw_model.iter780000.npz',
        'de_en_stt_raw_model.iter750000.npz',
        'de_en_stt_raw_model.iter720000.npz',
        'de_en_stt_raw_model.iter690000.npz',
        'de_en_stt_raw_model.iter660000.npz',
        'de_en_stt_raw_model.iter630000.npz',
        'de_en_stt_raw_model.iter600000.npz',
        'de_en_stt_raw_model.iter570000.npz',
        'de_en_stt_raw_model.iter540000.npz',
        'de_en_stt_raw_model.iter510000.npz',
        'de_en_stt_raw_model.iter480000.npz',
        'de_en_stt_raw_model.iter450000.npz',
        'de_en_stt_raw_model.iter420000.npz',
        'de_en_stt_raw_model.iter390000.npz',
        'de_en_stt_raw_model.iter360000.npz',
        'de_en_stt_raw_model.iter330000.npz',
        'de_en_stt_raw_model.iter300000.npz',
        'de_en_stt_raw_model.iter270000.npz',
        'de_en_stt_raw_model.iter240000.npz',
        'de_en_stt_raw_model.iter210000.npz',
        'de_en_stt_raw_model.iter180000.npz',
        'de_en_stt_raw_model.iter150000.npz',
        'de_en_stt_raw_model.iter120000.npz',
        'de_en_stt_raw_model.iter90000.npz',
        'de_en_stt_raw_model.iter60000.npz',
        'de_en_stt_raw_model.iter30000.npz']

    # stt_model_files = [
    #     'de_en_stt_model.iter1320000.npz',
    #     'de_en_stt_model.iter1290000.npz',
    #     'de_en_stt_model.iter1260000.npz',
    #     'de_en_stt_model.iter1230000.npz',
    #     'de_en_stt_model.iter1200000.npz',
    #     'de_en_stt_model.iter1170000.npz',
    #     'de_en_stt_model.iter1140000.npz',
    #     'de_en_stt_model.iter1110000.npz',
    #     'de_en_stt_model.iter1080000.npz',
    #     'de_en_stt_model.iter1050000.npz',
    #     'de_en_stt_model.iter1020000.npz',
    #     'de_en_stt_model.iter990000.npz',
    #     'de_en_stt_model.iter960000.npz',
    #     'de_en_stt_model.iter930000.npz',
    #     'de_en_stt_model.iter900000.npz',
    #     'de_en_stt_model.iter870000.npz',
    #     'de_en_stt_model.iter840000.npz',
    #     'de_en_stt_model.iter810000.npz',
    #     'de_en_stt_model.iter780000.npz',
    #     'de_en_stt_model.iter750000.npz',
    #     'de_en_stt_model.iter720000.npz',
    #     'de_en_stt_model.iter690000.npz',
    #     'de_en_stt_model.iter660000.npz',
    #     'de_en_stt_model.iter630000.npz',
    #     'de_en_stt_model.iter600000.npz',
    #     'de_en_stt_model.iter570000.npz',
    #     'de_en_stt_model.iter540000.npz',
    #     'de_en_stt_model.iter510000.npz',
    #     'de_en_stt_model.iter480000.npz',
    #     'de_en_stt_model.iter450000.npz',
    #     'de_en_stt_model.iter420000.npz',
    #     'de_en_stt_model.iter390000.npz',
    #     'de_en_stt_model.iter360000.npz',
    #     'de_en_stt_model.iter330000.npz',
    #     'de_en_stt_model.iter300000.npz',
    #     'de_en_stt_model.iter270000.npz',
    #     'de_en_stt_model.iter240000.npz',
    #     'de_en_stt_model.iter210000.npz',
    #     'de_en_stt_model.iter180000.npz',
    #     'de_en_stt_model.iter150000.npz',
    #     'de_en_stt_model.iter120000.npz',
    #     'de_en_stt_model.iter90000.npz',
    #     'de_en_stt_model.iter60000.npz',
    #     'de_en_stt_model.iter30000.npz']
    stt_model_files.reverse()

    bpe_model_files = [
        'de_en_bpe_raw_model.iter720000.npz',
        'de_en_bpe_raw_model.iter690000.npz',
        'de_en_bpe_raw_model.iter660000.npz',
        'de_en_bpe_raw_model.iter630000.npz',
        'de_en_bpe_raw_model.iter600000.npz',
        'de_en_bpe_raw_model.iter570000.npz',
        'de_en_bpe_raw_model.iter540000.npz',
        'de_en_bpe_raw_model.iter510000.npz',
        'de_en_bpe_raw_model.iter480000.npz',
        'de_en_bpe_raw_model.iter450000.npz',
        'de_en_bpe_raw_model.iter420000.npz',
        'de_en_bpe_raw_model.iter390000.npz',
        'de_en_bpe_raw_model.iter360000.npz',
        'de_en_bpe_raw_model.iter330000.npz',
        'de_en_bpe_raw_model.iter300000.npz',
        'de_en_bpe_raw_model.iter270000.npz',
        'de_en_bpe_raw_model.iter240000.npz',
        'de_en_bpe_raw_model.iter210000.npz',
        'de_en_bpe_raw_model.iter180000.npz',
        'de_en_bpe_raw_model.iter150000.npz',
        'de_en_bpe_raw_model.iter120000.npz',
        'de_en_bpe_raw_model.iter90000.npz',
        'de_en_bpe_raw_model.iter60000.npz',
        'de_en_bpe_raw_model.iter30000.npz'
    ]

    # bpe_model_files = ['de_en_bpe_model.iter660000.npz',
    #                    'de_en_bpe_model.iter630000.npz',
    #                    'de_en_bpe_model.iter600000.npz',
    #                    'de_en_bpe_model.iter570000.npz',
    #                    'de_en_bpe_model.iter540000.npz',
    #                    'de_en_bpe_model.iter510000.npz',
    #                    'de_en_bpe_model.iter480000.npz',
    #                    'de_en_bpe_model.iter450000.npz',
    #                    'de_en_bpe_model.iter420000.npz']
    # bpe_model_files = ['de_en_bpe_model.iter390000.npz',
    #                    'de_en_bpe_model.iter360000.npz',
    #                    'de_en_bpe_model.iter330000.npz',
    #                    'de_en_bpe_model.iter300000.npz',
    #                    'de_en_bpe_model.iter270000.npz',
    #                    'de_en_bpe_model.iter240000.npz',
    #                    'de_en_bpe_model.iter210000.npz',
    #                    'de_en_bpe_model.iter180000.npz',
    #                    'de_en_bpe_model.iter150000.npz',
    #                    'de_en_bpe_model.iter120000.npz',
    #                    'de_en_bpe_model.iter90000.npz',
    #                    'de_en_bpe_model.iter60000.npz',
    #                    'de_en_bpe_model.iter30000.npz']
    bpe_model_files.reverse()

    # stt_prefix = '/home/nlp/aharonr6/git/research/nmt/models/de_en_stt/overtime'
    # bpe_prefix = '/home/nlp/aharonr6/git/research/nmt/models/de_en_bpe/overtime'

    stt_prefix = '/home/nlp/aharonr6/git/research/nmt/models/de_en_stt_raw/overtime'
    bpe_prefix = '/home/nlp/aharonr6/git/research/nmt/models/de_en_bpe_raw/overtime'

    print 'bpe:\n'
    for filepath in bpe_model_files:
        alignment_path = '{}/{}_dev_alignments.txt'.format(bpe_prefix, filepath)
        scores = get_distortion_from_alignments_file(alignment_path)
        avg = numpy.sum(scores)/len(scores)
        print filepath, '\t', avg

    print 'stt:\n'
    for filepath in stt_model_files:
        alignment_path = '{}/{}_dev_alignments.txt'.format(stt_prefix, filepath)
        scores = get_distortion_from_alignments_file(alignment_path)
        avg = numpy.sum(scores)/len(scores)
        print filepath, '\t', avg



def get_distortion_from_alignments_file(filepath):
    count = 0
    scores = []
    with codecs.open(filepath, 'r') as file1:

        while (file1):
            count += 1
            # print count
            sid, mma, target_labels, source_labels = read_alignment_matrix(file1)
            if not target_labels:
                break

            no_tree_matrix, no_tree_target_labels = strip_tree(mma, target_labels)

            tree_score = get_diagonal_subsequent_reordering_score(no_tree_matrix, source_labels, no_tree_target_labels)
            scores.append(tree_score)
            line = file1.readline()
            if not line:
                break
    return scores


def main(file1, file2):
    # compare_sentence_level_bleu()
    distortion_over_time()
    # inspect_alignment_matrices(file1, file2)

    # cnt1, cnt2 = get_distortion_step_sizes(file1, file2)
    # cnt1 = sorted(cnt1.items(), key=itemgetter(0))
    # cnt2 = sorted(cnt2.items(), key=itemgetter(0))

    # distortion_over_time()

    # subtrees = get_subtrees_with_reordering(file1)
    # sorted_subtrees = sorted(subtrees, key=itemgetter(0), reverse=True)

    return


if __name__ == "__main__":
    main('bla','bla')
    parser = argparse.ArgumentParser()
    # '/Users/mnadejde/Documents/workspace/MTMA2016/models/wmt16_systems/en-de/test.alignment'
    parser.add_argument('--input', '-i', type=argparse.FileType('r'),
                        default='~/git/research/nmt/models/de_en_stt/dev_alignments.txt.best',
                        metavar='PATH',
                        help="Input file (default: standard input)")

    parser.add_argument('--comp', '-c', type=argparse.FileType('r'),
                        default='~/git/research/nmt/models/de_en_wmt16/dev_alignments.txt',
                        metavar='PATH',
                        help="2nd Input file (default: standard input)")

    args = parser.parse_args()

    main(args.input, args.comp)
