from collections import OrderedDict
from typing import Iterable, Tuple


def get_word_idx(sens: Iterable[str], sos_token: int = 0, eos_token: int = 1) -> Tuple[OrderedDict, OrderedDict]:
    """get word2idx and idx2word"""
    word2idx = OrderedDict({"SOS": sos_token, "EOS": eos_token})
    idx2word = OrderedDict({sos_token: "SOS", eos_token: "EOS"})
    n_words = 2  # Count SOS and EOS

    for sen in sens:
        for word in sen.split(" "):
            if word not in word2idx:
                word2idx[word] = n_words
                idx2word[n_words] = word
                n_words += 1
    return word2idx, idx2word
