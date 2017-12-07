from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from pprint import pprint
import random
import re
from cPickle import dump, load

LINE_PAT = re.compile(r'([0-9])*[ \t](.*)[ \t]{(.*)}\Z')
QUOTE_PAT = re.compile(r'"(.*)"')
GENRE_CT = 10
VOCAB_CT = 16188
START_IDX = 0
END_IDX = VOCAB_CT - 1
UNK_TOK = '<unk>'

DIR_NAME = 'mult_hot_out/'
DATA_FILE_PATH = '../summaries_genre.txt'
TEST_DATA_FILE_PATH = '../summaries_genre_short.txt'
REF_VOCAB_PATH = 'refs/ref_vocab.dict'

TRAIN_PCT = 0.99
PAD = 16


def main(test=False):

    data_path = DATA_FILE_PATH if not test else TEST_DATA_FILE_PATH

    def get_genre(item):
        prelim = item.split(': ')
        if len(prelim) < 2:
            return ""
        result = re.match(QUOTE_PAT, prelim[1])
        if result is not None:
            return result.group(1)
        return ""

    genre_cts = Counter()
    genres = None

    line_ct = 0
    with open(data_path) as in_file:
        for line in in_file:
            line = line.decode('utf-8', 'ignore')
            matches = re.match(LINE_PAT, line.rstrip())
            genre_cts += Counter(list(map(get_genre, matches.group(3).split(', '))))
        genres = genre_cts.most_common(GENRE_CT)
        print(genres)
        genres = frozenset(map(lambda x: x[0], genres))

        in_file.seek(0)
        tmp_genre = {k: [0, 0] for k in genres}
        with open(DIR_NAME + 'truncated_summaries_genre.txt', 'w') as trunc_file:
            for line in in_file:
                line = line.decode('utf-8', 'ignore')
                line = line.encode('ascii', 'ignore')
                matches = re.match(LINE_PAT, line.rstrip())
                cur_genres = frozenset(map(get_genre, matches.group(3).split(', ')))
                if not cur_genres.isdisjoint(genres):
                    intersect = cur_genres.intersection(genres)
                    for k in intersect:
                        tmp_genre[k][0] += 1
                    sentences = sent_tokenize(matches.group(2))
                    for sentence in sentences:
                        if len(word_tokenize(sentence)) <= PAD:
                            for k in intersect:
                                tmp_genre[k][1] += 1
                            trunc_file.write(matches.group(1) + '\t' + sentence + '\t{' + matches.group(3) + '}\n')
                            line_ct += 1
        print('Genre summary and line counts:')
        pprint(tmp_genre)

    genre2idx = {genre: idx for idx, genre in enumerate(genres)}

    vocab2idx = None
    with open(REF_VOCAB_PATH, 'rb') as ref_file:
        vocab2idx = load(ref_file)

    with open(DIR_NAME + 'truncated_summaries_genre.txt') as in_file:
        with open(DIR_NAME + 'vae_data/train.txt', 'w') as vtr, \
                open(DIR_NAME + 'vae_data/test.txt', 'w') as vte, \
                open(DIR_NAME + 'disc_data/train.txt', 'w') as dtr, \
                open(DIR_NAME + 'disc_data/test.txt', 'w') as dte:
            def embed(token):
                if token in vocab2idx:
                    return str(vocab2idx[token])
                else:
                    return str(vocab2idx[UNK_TOK])
            randbits = [True for x in range(int(0.99 * line_ct))] + \
                [False for x in range(line_ct - int(0.99 * line_ct))]
            counter = 0
            for line in in_file:
                matches = re.match(LINE_PAT, line.rstrip())
                cur_genres = set(map(get_genre, matches.group(3).split(', '))) & genres
                emb_genres = sorted(list(map(lambda x: genre2idx[x], cur_genres)))
                emb_genres = list(map(str, emb_genres))

                tokens = word_tokenize(matches.group(2).rstrip().lower())
                emb_tokens = list(map(embed, tokens))
                emb_tokens = emb_tokens + [str(END_IDX) for x in range(PAD - len(emb_tokens))]

                disc_out = dtr
                vae_out = vtr
                if not randbits[counter]:
                    disc_out = dte
                    vae_out = vte

                counter += 1
                disc_out.write(' '.join(emb_genres) + ',' + ' '.join(emb_tokens) + '\n')
                vae_out.write(' '.join(emb_tokens) + '\n')

    with open(DIR_NAME + 'vocab.dict', 'wb') as vocab_f:
        dump(vocab2idx, vocab_f)
    with open(DIR_NAME + 'genre.dict', 'wb') as genre_f:
        dump(genre2idx, genre_f)


if __name__ == '__main__':
    main()
