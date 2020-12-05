import torch
from clean_data import normalizeString
import random
from typing import Tuple, List, Union
from util import get_word_idx
from collections import OrderedDict

ENG_PREFIXES = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


class TranslationDataSet():
    def __init__(self, data_path: str, src_lang_name: str = "src", tar_lang_name: str = "tar", sos_token: int = 0,
                 eos_token: int = 1, max_length: int = 10, is_ch: bool = False, shuffle: bool = True,
                 reverse_pair: bool = True):
        self.data_path = data_path
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_length = max_length
        self.src_lang_name = src_lang_name
        self.tar_lang_name = tar_lang_name
        self.is_ch = is_ch
        self.reverse_pair = reverse_pair
        self.shuffle = shuffle
        # get word idx and pairs
        self.pairs, self.src_word2idx, self.src_idx2word, self.tar_word2idx, self.tar_idx2word = self.get_pairs_and_word_idx()
        if self.shuffle:
            random.shuffle(self.pairs)
        self.data_iter = iter(self.pairs)
        self.num_steps_per_epoch = len(self.pairs)

    def get_pairs_and_word_idx(self) -> Tuple[
        List[List[str]], OrderedDict, OrderedDict, OrderedDict, OrderedDict]:
        """  获取统计字数的模型和句子对 """
        # Read the file and split into lines
        pairs = []
        with open(self.data_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.strip().split("\t")
                if len(ss) == 2:
                    if self.reverse_pair:
                        ss.reverse()
                    pairs.append([normalizeString(s, is_ch=self.is_ch) for s in ss])
        # filter pairs
        if self.is_ch:
            pairs = [pair for pair in pairs if
                     len(pair[0].split(' ')) < self.max_length
                     and len(pair[1].split(' ')) < self.max_length]
        else:
            pairs = [pair for pair in pairs if
                     len(pair[0].split(' ')) < self.max_length
                     and len(pair[1].split(' ')) < self.max_length
                     and pair[1].startswith(ENG_PREFIXES)]
        print("num pairs:{}".format(len(pairs)))
        # get word idx
        src_word2idx, src_idx2word = get_word_idx([p[0] for p in pairs], self.sos_token, self.eos_token)
        tar_word2idx, tar_idx2word = get_word_idx([p[1] for p in pairs], self.sos_token, self.eos_token)
        return pairs, src_word2idx, src_idx2word, tar_word2idx, tar_idx2word

    def get_batch_tensor_data(self) -> Union[Tuple[torch.LongTensor, torch.LongTensor], None]:
        """ here, assuming that batch size is 1 """
        try:
            pair = next(self.data_iter)
        except StopIteration:
            return None
        src_sen, tar_sen = pair
        # 1*seq_len
        src_tensor = torch.LongTensor([[self.src_word2idx[word] for word in src_sen.split(" ")] + [self.eos_token]])
        tar_tensor = torch.LongTensor([[self.tar_word2idx[word] for word in tar_sen.split(" ")] + [self.eos_token]])

        return src_tensor, tar_tensor

    def __iter__(self):
        return self

    def __next__(self):
        tensor_data = self.get_batch_tensor_data()
        if tensor_data is None:  # finished one epoch
            if self.shuffle:
                random.shuffle(self.pairs)
                self.data_iter = iter(self.pairs)
            raise StopIteration
        else:
            return tensor_data


if __name__ == "__main__":
    data = TranslationDataSet(data_path="data/lccc_chat.txt", is_ch=True, max_length=15)
    for epoch in range(1):
        for step, (t1, t2) in enumerate(data):
            print(1)
