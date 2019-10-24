from keras_preprocessing.sequence import pad_sequences

k = 0
stride = 0


def set_k_and_stride(set_k, set_stride):
    global k
    k = set_k
    global stride
    stride = set_stride


def transform_data(data, max_length):
    # data = shuffle(data)
    x_train_list = list(map(seq_to_kmer, [item[0] for item in data]))
    x_train_padded = pad_sequences(x_train_list, maxlen=max_length, padding='post')
    y_train = [float(item[1]) for item in data]
    return x_train_padded, y_train


# Preprocessing, transforming a sequence of nucleotides to a a set of overlapping k-mers, and these k-mers to a vector
def seq_to_kmer(seq):
    char_to_number = {"A": 1, "T": 2, "C": 3, "G": 4, "N": 0}
    position = 0
    kmers = []
    while position < len(seq):
        if position + k > len(seq):
            kmer = (seq[position:(position + k)] + k * 'N')[:k]
        else:
            kmer = (seq[position:(position + k)])
        enc = 0
        for i, c in enumerate(kmer):
            enc = enc + pow(5, (k - i - 1)) * char_to_number[c]
        kmers.append(enc)
        position = position + stride
    return kmers
