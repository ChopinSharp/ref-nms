from tqdm import tqdm
import numpy as np

from lib.refer import REFER


GLOVE_WORD_NUM = 2196017
GLOVE_FILE = 'data/glove.840B.300d.txt'
VOCAB_THRESHOLD = 2
VOCAB_SAVE_PATH = 'cache/std_vocab_{}_{}.txt'
GLOVE_SAVE_PATH = 'cache/std_glove_{}_{}.npy'


def load_glove_feats():
    glove_dict = {}  # from word of <str> to vector of <generator>
    with open(GLOVE_FILE, 'r') as f:
        with tqdm(total=GLOVE_WORD_NUM, desc='Loading GloVe', ascii=True) as pbar:
            for line in f:
                tokens = line.split(' ')
                assert len(tokens) == 301
                word = tokens[0]
                vec = list(map(lambda x: float(x), tokens[1:]))
                glove_dict[word] = vec
                pbar.update(1)
    return glove_dict


def build_vocabulary(dataset, split_by, glove_dict):
    # load refer
    refer = REFER('data/refer', dataset, split_by)

    # filter corpus by frequency and GloVe
    word_count = {}
    for ref in refer.Refs.values():
        for sent in ref['sentences']:
            for word in sent['tokens']:
                word_count[word] = word_count.get(word, 0) + 1
    vocab, typo, rare = [], [], []
    for wd, n in word_count.items():
        if n < VOCAB_THRESHOLD:
            rare.append(wd)
        else:
            if wd in glove_dict:
                vocab.append(wd)
            else:
                typo.append(wd)
    assert len(vocab) + len(typo) + len(rare) == len(word_count)
    rare_count = sum([word_count[wd] for wd in rare])
    typo_count = sum([word_count[wd] for wd in typo])
    total_words = sum(word_count.values())
    print('number of good words: {}'.format(len(vocab)))
    print('number of rare words: {}/{} = {:.2f}%'.format(
        len(rare), len(word_count), len(rare)*100/len(word_count)))
    print('number of typo words: {}/{} = {:.2f}%'.format(
        len(typo), len(word_count), len(typo)*100/len(word_count)))
    print('number of UNKs in sentences: ({}+{})/{} = {:.2f}%'.format(
        rare_count, typo_count, total_words, (rare_count+typo_count)*100/total_words))

    # sort vocab and construct glove feats
    vocab = sorted(vocab)
    vocab_glove = []
    for wd in vocab:
        vocab_glove.append(glove_dict[wd])
    vocab.insert(0, '<unk>')
    vocab_glove.insert(0, [0.] * 300)
    vocab_glove = np.array(vocab_glove, dtype=np.float32)

    # save vocab and glove feats
    vocab_save_path = VOCAB_SAVE_PATH.format(dataset, split_by)
    glove_save_path = GLOVE_SAVE_PATH.format(dataset, split_by)
    print('saving vacob in {}'.format(vocab_save_path))
    with open(vocab_save_path, 'w') as f:
        for wd in vocab:
            f.write(wd + '\n')
    print('saving vocab glove in {}'.format(glove_save_path))
    np.save(glove_save_path, vocab_glove)


def main():
    print('building vocab...')
    glove_feats = load_glove_feats()
    for dataset, split_by in [('refcoco', 'unc'), ('refcoco+', 'unc'), ('refcocog', 'umd')]:
        print('building {}_{}...'.format(dataset, split_by))
        build_vocabulary(dataset, split_by, glove_feats)
    print()


main()
