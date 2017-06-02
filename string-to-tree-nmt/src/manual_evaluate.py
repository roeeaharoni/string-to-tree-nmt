# todo: print side by side: reference (tree+sent), bpe2tree (tree+sent), bpe2bpe (5sent)
# todo: print attention viz (for both bpe2tree, bpe2bpe)
# todo: measure amount of syntactic structures, tree depth,
# we want to understand where the trees help and where they damage

import codecs
def main():
    src_path = '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-src.de'
    ref_path = '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-ref.en'
    ref_trees_path = '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-ref.tok.true.parsed.en.bped'
    bpe2tree_sents_path = '../models/de_en_stt/newstest2015-deen-src.tok.true.de.bpe.output.sents.dev.postprocessed.best'
    bpe2tree_trees_path = '../models/de_en_stt/newstest2015-deen-src.tok.true.de.bpe.output.trees.dev'
    bpe2bpe_sents_path = '/Users/roeeaharoni/git/research/nmt/models/de_en_wmt16/newstest2015-deen-src.de.output.en'
    output_path = '../models/de_en_stt/dev_comparison.txt'

    i = 0
    with codecs.open(output_path, 'w', encoding='utf-8') as output_file:
        with codecs.open(src_path, 'r', encoding='utf-8') as src_file:
            with codecs.open(ref_path, 'r', encoding='utf-8') as ref_file:
                with codecs.open(ref_trees_path, 'r', encoding='utf-8') as ref_trees_file:
                    with codecs.open(bpe2tree_sents_path, 'r', encoding='utf-8') as bpe2tree_sents_file:
                        with codecs.open(bpe2tree_trees_path, 'r', encoding='utf-8') as bpe2tree_trees_file:
                            with codecs.open(bpe2bpe_sents_path, 'r', encoding='utf-8') as bpe2bpe_sents_file:
                                while True:

                                    src_sent = src_file.readline()
                                    ref_sent = ref_file.readline()
                                    ref_tree = ref_trees_file.readline()
                                    bpe2tree_sent = bpe2tree_sents_file.readline()
                                    bpe2tree_tree = bpe2tree_trees_file.readline()
                                    bpe2bpe_sent = bpe2bpe_sents_file.readline()

                                    if not ref_sent:
                                        break

                                    format_str = u'line id: {}\n\nsrc:\n{}\nref:\n{}\nbpe2tree:\n{}\nbpe2bpe:\n{}\n\n'
                                    print format_str.format(
                                        i, src_sent,
                                        ref_sent, bpe2tree_sent, bpe2bpe_sent)

                                    output_file.write(format_str.format(
                                        i, src_sent,
                                        ref_sent, bpe2tree_sent, bpe2bpe_sent))

                                    i += 1


    return

if __name__ == '__main__':
    main()