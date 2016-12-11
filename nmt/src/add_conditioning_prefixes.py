import codecs
from collections import defaultdict


def main ():
    # src_file = '/Users/roeeaharoni/git/research/nmt/en-he/IWSLT14.TED.dev2010.he-en.en.tok'
    # target_file = '/Users/roeeaharoni/git/research/nmt/en-he/IWSLT14.TED.dev2010.he-en.en.tok.output.dev'
    src_file = "/Users/roeeaharoni/git/research/nmt/en-he/train.tags.he-en.en.tok"
    target_file = "/Users/roeeaharoni/git/research/nmt/en-he/train.tags.he-en.he.tok"
    output_file = "/Users/roeeaharoni/git/research/nmt/en-he/train.tags.he-en.en.tok.len"

    new_src_lines = []
    len_histogram = defaultdict(int)
    binned_histogram = defaultdict(int)

    with codecs.open(src_file, encoding='utf8') as src:
        with codecs.open(target_file, encoding='utf8') as trgt:
            print 'tokenizing...'
            src_lines = src.readlines()
            trgt_lines = trgt.readlines()
            for i, line in enumerate(trgt_lines):
                tokens = line.split()
                length = len(tokens)
                len_histogram[length] = len_histogram[length] + 1
                binned_length = length + 5 - length % 5
                binned_histogram[binned_length] += 1
                new_src_lines.append('TL{} '.format(binned_length) + src_lines[i])

    for i in xrange(100):
        print u'{}\n{}'.format(new_src_lines[i], trgt_lines[i])

    with codecs.open(output_file, 'w', encoding='utf8') as predictions:
        for i, line in enumerate(new_src_lines):
            predictions.write(u'{}'.format(line))

    for key in len_histogram:
        print key, '\t', len_histogram[key]

    for key in binned_histogram:
        print key, '\t', binned_histogram[key]




if __name__ == '__main__':
    main()