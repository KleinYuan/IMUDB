"""
Below are mostly borrowed from https://github.com/dapowan/LIMU-BERT-Public/blob/master/utils.py
"""
import numpy as np


def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    # alpha = 6
    # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)# possion
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)


def gather(data, position1, position2):
    result = []
    for i in range(position1.shape[0]):
        result.append(data[position1[i], position2[i]])
    return np.array(result)


def mask(data, position1, position2):
    for i in range(position1.shape[0]):
        data[position1[i], position2[i]] = np.zeros(position2[i].size)
    return data


def replace(data, position1, position2):
    for i in range(position1.shape[0]):
        data[position1[i], position2[i]] = np.random.random(position2[i].size)
    return data


def create_mask(example, mask_ratio, max_gram, mask_prob, replace_prob):
    shape = example.shape

    # the number of prediction is sometimes less than max_pred when sequence is short
    n_pred = max(1, int(round(shape[0] * mask_ratio)))

    # For masked Language Models
    # mask_pos = bert_mask(shape[0], n_pred)
    mask_pos = span_mask(shape[0], max_gram, goal_num_predict=n_pred)

    instance_mask = example.copy()

    if isinstance(mask_pos, tuple):
        mask_pos_index = mask_pos[0]
        if np.random.rand() < mask_prob:
            mask(instance_mask, mask_pos[0], mask_pos[1])
        elif np.random.rand() < replace_prob:
            replace(instance_mask, mask_pos[0], mask_pos[1])
    else:
        mask_pos_index = mask_pos
        if np.random.rand() < mask_prob:
            instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
        elif np.random.rand() < replace_prob:
            instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))
    seq = example[mask_pos_index, :]
    return instance_mask, np.array(mask_pos_index), np.array(seq)
