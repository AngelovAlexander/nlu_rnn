import sys
from utils import *

if __name__ == "__main__":

    train_size = 10000
    dev_size = 1000
    vocab_size = 2000
    data_folder = sys.argv[1]

    vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0,
                              names=['count', 'freq'], )
    #num_to_word = dict(enumerate(vocab.index[:vocab_size]))
    #word_to_num = invert_dict(num_to_word)
    
    distances = load_subj_and_verb_ind(data_folder + '/wiki-train.txt')
    #S_train = docs_to_indices(sents, word_to_num, 0, 0)
    #X_train, D_train = seqs_to_npXY(S_train)
    indices, counts = np.unique(distances, return_counts=True)
    print(indices, counts)

    distances = load_subj_and_verb_ind(data_folder + '/wiki-dev.txt')
    #S_train = docs_to_indices(sents, word_to_num, 0, 0)
    #X_train, D_train = seqs_to_npXY(S_train)
    indices, counts = np.unique(distances, return_counts=True)
    print(indices, counts)
    sent = load_lm_np_dataset_verbs(data_folder + '/wiki-train.txt')
    a, b = np.unique(sent, axis=0, return_counts=True)
    print(a[26])
    print(b[26])
    print(a[246])
    print(b[246])


    """
    sent1, sent2 = load_np_two_datasets(data_folder + '/wiki-train.txt', 1)
    S1_train = docs_to_indices(sent1, word_to_num, 0, 0)
    X1_train, D1_train = seqs_to_npXY(S1_train)

    X1_train = X1_train[:train_size]
    Y1_train = D1_train[:train_size]
    """