from collections import Counter
from nltk.tokenize import word_tokenize
import re
from cPickle import dump, load

LINE_PAT = re.compile(r'[0-9]*[ \t](.*)[ \t]{(.*)}\Z')
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

    with open(data_path) as in_file:
        for line in in_file:
            line = line.decode('utf-8', 'ignore')
            matches = re.match(LINE_PAT, line.rstrip())
            genre_cts += Counter(list(map(get_genre, matches.group(2).split(', '))))
        genres = genre_cts.most_common(GENRE_CT)
        print(genres)
        genres = frozenset(map(lambda x: x[0], genres))

        in_file.seek(0)
        with open(DIR_NAME + 'truncated_summaries_genre.txt', 'w') as trunc_file:
            for line in in_file:
                line = line.decode('utf-8', 'ignore')
                line = line.encode('ascii', 'ignore')
                matches = re.match(LINE_PAT, line.rstrip())
                cur_genres = frozenset(map(get_genre, matches.group(2).split(', ')))
                if not cur_genres.isdisjoint(genres):
                    trunc_file.write(line)



    genre2idx = {genre: idx for idx, genre in enumerate(genres)}


    vocab2idx = None
    with open(REF_VOCAB_PATH, 'rb') as ref_file:
        vocab2idx = load(ref_file)


    with open(DIR_NAME + 'truncated_summaries_genre.txt') as in_file:
        with open(DIR_NAME + 'summaries_genre_embed.txt', 'w') as out_file:
            def embed(token):
                if token in vocab2idx:
                    return str(vocab2idx[token])
                else:
                    return str(vocab2idx[UNK_TOK])

            for line in in_file:
                matches = re.match(LINE_PAT, line.rstrip())
                cur_genres = set(map(get_genre, matches.group(2).split(', '))) & genres
                emb_genres = sorted(list(map(lambda x: genre2idx[x], cur_genres)))
                emb_genres = list(map(str, emb_genres))

                tokens = word_tokenize(matches.group(1).rstrip().lower())
                emb_tokens = [str(START_IDX)] + list(map(embed, tokens)) + [str(END_IDX)]
                out_file.write(' '.join(emb_genres) + ',' + ' '.join(emb_tokens) + '\n')

    with open(DIR_NAME + 'vocab.dict', 'wb') as vocab_f:
        dump(vocab2idx, vocab_f)
    with open(DIR_NAME + 'genre.dict', 'wb') as genre_f:
        dump(genre2idx, genre_f)


if __name__ == '__main__':
    main()
