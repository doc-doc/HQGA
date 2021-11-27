from build_vocab import Vocabulary
from utils import *
import numpy as np
import random as rd
rd.seed(0)

def word2vec(vocab, glove_file, save_filename):
    glove = load_file(glove_file)
    word2vec = {}
    for line in glove:
        line = line.split(' ')
        word2vec[line[0]] = np.array(line[1:]).astype(np.float32)

    temp = []
    for word, vec in word2vec.items():
        temp.append(vec)
    temp = np.asarray(temp)
    row, col = temp.shape
    # print(row, col)
    pad = np.mean(temp, axis=0)
    start = np.mean(temp[:int(row//2), :], axis=0)
    end = np.mean(temp[int(row//2):, :], axis=0)
    special_tokens = [pad, start, end]
    count = 0
    bad_words = []
    sort_idx_word = sorted(vocab.idx2word.items(), key=lambda k:k[0])
    glove_embed = np.zeros((len(vocab), 300))
    for row, item in enumerate(sort_idx_word):
        idx, word = item[0], item[1]
        if word in word2vec:
            glove_embed[row] = word2vec[word]
        else:
            if row < 3:
                glove_embed[row] = special_tokens[row]
            else:
                glove_embed[row] = np.random.randn(300)*0.4
            print(word)
            bad_words.append(word)
            count += 1
    print(glove_embed.shape)
    # save_file(bad_words, 'bad_words.json')
    np.save(save_filename, glove_embed)
    print(count)

def main():
    dataset, task = 'msvd', ''
    data_dir = f'dataset/{dataset}/{task}/'
    vocab_file = osp.join(data_dir, 'vocab.pkl')
    vocab = pkload(vocab_file)
    glove_file = '../data/glove.840B.300d.txt'
    save_filename = f'dataset/{dataset}/{task}/glove_embed.npy'
    word2vec(vocab, glove_file, save_filename)

if __name__ == "__main__":
    main()