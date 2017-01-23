import numpy
import matplotlib.pyplot as plt
import sys
import os
import json
import argparse
import math
import codecs
from collections import defaultdict


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
def plot_heat_map(mma, target_labels, source_labels, mma2=None, target_labels2=None, source_labels2=None, id=None):
    if mma2 == None:
        fig, ax = plt.subplots()
    else:
        score = get_diagonal_subsequent_reordering_score(mma, source_labels, target_labels)
        plt.figure(figsize=(16, 8))
        ax = plt.subplot(121)
        ax.text(float(len(source_labels)) / 2, len(target_labels) + 1, 'bpe2tree:{}'.format(score))

    heatmap = ax.pcolor(mma, cmap=plt.cm.Blues)

    # put the major ticks at the middle of each cell
    ax.set_xticks(numpy.arange(mma.shape[1]) + 0.5, minor=False)
    ax.set_yticks(numpy.arange(mma.shape[0]) + 0.5, minor=False)

    # without this I get some extra columns rows
    # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
    ax.set_xlim(0, int(mma.shape[1]))
    ax.set_ylim(0, int(mma.shape[0]))

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # source words -> column labels
    ax.set_xticklabels(source_labels, minor=False)
    # target words -> row labels
    ax.set_yticklabels(target_labels, minor=False)

    plt.xticks(rotation=45)

    if mma2 is not None:
        score = get_diagonal_subsequent_reordering_score(mma2, source_labels2, target_labels2)
        ax2 = plt.subplot(122)
        ax2.text(float(len(source_labels2)) / 2, len(target_labels2) + 1, 'bpe2bpe:{}'.format(score))
        heatmap = ax2.pcolor(mma2, cmap=plt.cm.Blues)

        # put the major ticks at the middle of each cell
        ax2.set_xticks(numpy.arange(mma2.shape[1]) + 0.5, minor=False)
        ax2.set_yticks(numpy.arange(mma2.shape[0]) + 0.5, minor=False)

        # without this I get some extra columns rows
        # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
        ax2.set_xlim(0, int(mma2.shape[1]))
        ax2.set_ylim(0, int(mma2.shape[0]))

        # want a more natural, table-like display
        ax2.invert_yaxis()
        ax2.xaxis.tick_top()

        # source words -> column labels
        ax2.set_xticklabels(source_labels2, minor=False)
        # target words -> row labels
        ax2.set_yticklabels(target_labels2, minor=False)

        plt.xticks(rotation=45)

    # plt.tight_layout()
    plt.show()
    if id:
        plt.savefig('plots/{}.png'.format(id-1))


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
def inspect_alignment_matrices(first, second=None):
    count = 0
    syntax_based_higher = 0
    bpe_based_higher = 0
    high_bpe = defaultdict(int)
    high_syntax = defaultdict(int)
    while (first):
        count += 1
        print count
        sid, mma, target_labels, source_labels = read_alignment_matrix(first)
        if not target_labels:
            break

        # remove rows of tree symbols
        rows = []
        for i, t in enumerate(target_labels):
            if u'(' not in t and u')' not in t and i < mma.shape[0]:
                try:
                    rows.append(mma[i, :])
                except Exception as e:
                    print 'bla'

        no_tree_matrix = numpy.array(rows)
        no_tree_target_labels = [l for l in target_labels if u'(' not in l and u')' not in l]
        tree_score = get_diagonal_subsequent_reordering_score(no_tree_matrix, source_labels, no_tree_target_labels)

        # compare to WMT model
        if second:
            sid2, mma2, target_labels2, source_labels2 = read_alignment_matrix(second)
            if target_labels2 is None:
                print 'no target labels!'

            bpe_score = get_diagonal_subsequent_reordering_score(mma2, source_labels2, target_labels2)

        # if tree_score > 4:
        plot_heat_map(no_tree_matrix, no_tree_target_labels, source_labels, mma2, target_labels2, source_labels2, count)

        if bpe_score < tree_score:
            syntax_based_higher += 1
        else:
            bpe_based_higher += 1
            # (target_len, source_len) = no_tree_matrix.shape

        for k in xrange(15):
            if tree_score == 0:
                print 'neg'
            if bpe_score >= k:
                high_bpe[k] += 1
            if tree_score >= k:
                high_syntax[k] += 1


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
        line = first.readline()
        if not line:
            break
        line2 = second.readline()
        if not line2:
            break

    print 'syntax had more reordering in: ', syntax_based_higher
    print 'bpe had more reordering in: ', bpe_based_higher

    print 'bpe high'
    for k in xrange(15):
        print k, '\t', high_bpe[k]
    print 'tree high'
    for k in xrange(15):
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
        '/Users/roeeaharoni/git/research/nmt/models/de_en_wmt16/newstest2015-deen-src.de.output.en')

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
                      'bpe_score': scores_bpe[i][2]})

    # large positive diffs = bpe is better
    out_format = u'id: {}\n\ndiff: {}\n\nsrc:\n{}\nref:\n{}\nstripped tree ({}):\n{}\ntree: {}\nbpe ({}):\n{} \n\n'

    comparison_path = 'comparison.txt'
    with codecs.open(comparison_path, 'w', 'utf-8') as comparison_file:
        for i, d in enumerate(diffs):
            output = out_format.format(i,
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

    # diffs.sort(key=lambda x: x['diff'])
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


def main(input, comp):
    compare_sentence_level_bleu()

    inspect_alignment_matrices(input, comp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # '/Users/mnadejde/Documents/workspace/MTMA2016/models/wmt16_systems/en-de/test.alignment'
    parser.add_argument('--input', '-i', type=argparse.FileType('r'),
                        default='/Users/mnadejde/Documents/workspace/MTMA2016/models/wmt16_systems/ro-en/newstest2016-roen-src.ro.alignment',
                        metavar='PATH',
                        help="Input file (default: standard input)")

    parser.add_argument('--comp', '-c', type=argparse.FileType('r'),
                        metavar='PATH',
                        help="2nd Input file (default: standard input)")

    args = parser.parse_args()

    main(args.input, args.comp)
