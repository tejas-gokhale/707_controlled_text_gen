from collections import Counter
from nltk.tokenize import word_tokenize
import re
from pickle import dump

LINE_PAT = re.compile(r'[0-9]*[ \t](.*)[ \t]{(.*)}\Z')
QUOTE_PAT = re.compile(r'"(.*)"')
GENRE_CT = 10
VOCAB_CT = 16188
START_TOK = '<START>'
END_TOK = '<END>'
UNK_TOK = '<UNK>'

DIR_NAME = 'mult_hot_out/'
DATA_FILE_PATH = '../summaries_genre.txt'
TEST_DATA_FILE_PATH = '../summaries_genre_short.txt'


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
    vocab_cts = Counter()

    with open(data_path, errors='ignore') as in_file:
        for line in in_file:
            matches = re.match(LINE_PAT, line.rstrip())
            genre_cts += Counter(list(map(get_genre, matches.group(2).split(', '))))
        genres = genre_cts.most_common(GENRE_CT)
        print(genres)
        genres = frozenset(map(lambda x: x[0], genres))

        in_file.seek(0)

        for line in in_file:
            matches = re.match(LINE_PAT, line.rstrip())
            cur_genres = frozenset(map(get_genre, matches.group(2).split(', ')))
            if not cur_genres.isdisjoint(genres):
                vocab_cts += Counter(word_tokenize(matches.group(1).lower()))

        vocab = vocab_cts.most_common(VOCAB_CT - 3)
        print(vocab[:50])
        vocab2idx = {word: idx for idx, (word, _) in enumerate(vocab)}
        length = len(vocab2idx)
        vocab2idx[START_TOK] = length
        vocab2idx[END_TOK] = length + 1
        vocab2idx[UNK_TOK] = length + 2


        in_file.seek(0)

        with open(DIR_NAME + 'truncated_summaries_genre.txt', 'w') as trunc_file:
            for line in in_file:
                line = list(map(lambda x: x if x in vocab2idx else UNK_TOK, word_tokenize(line.lower())))
                trunc_file.write(' '.join(line))

    with open(DIR_NAME + 'truncated_summaries_genre.txt', 'w') as trunc_file:
        for line in in_file:
            line = list(map(lambda x: x if x in vocab2idx else UNK_TOK, word_tokenize(line.lower())))
            trunc_file.write(' '.join(line))


    genre2idx = {genre: idx for idx, genre in enumerate(genres)}

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
                emb_tokens = [str(vocab2idx[START_TOK])] + list(map(embed, tokens)) + [str(vocab2idx[END_TOK])]
                out_file.write(' '.join(emb_genres) + ',' + ' '.join(emb_tokens) + '\n')

    with open(DIR_NAME + 'vocab.dict', 'wb') as vocab_f:
        dump(vocab2idx, vocab_f)
    with open(DIR_NAME + 'genre.dict', 'wb') as genre_f:
        dump(genre2idx, genre_f)


if __name__ == '__main__':
    main(True)
