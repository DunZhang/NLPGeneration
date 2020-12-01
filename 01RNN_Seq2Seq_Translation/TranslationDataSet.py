import torch
from clean_data import normalizeString
import random
from typing import Tuple, List, Union


class WordInfo:
    def __init__(self, name, sos_token: int = 0, eos_token: int = 1):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {sos_token: "SOS", eos_token: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class TranslationDataSet():
    def __init__(self, data_path: str, src_lang_name: str = "src", tar_lang_name: str = "tar", sos_token: int = 0,
                 eos_token: int = 1,
                 max_length: int = 10, is_ch: bool = False, shuffle: bool = True, batch_size: int = 1):
        self.data_path = data_path
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_length = max_length
        self.src_lang_name = src_lang_name
        self.tar_lang_name = tar_lang_name
        self.is_ch = is_ch
        self.shuffle = shuffle
        self.batch_size = batch_size
        # 获取word info和pairs
        self.input_lang, self.output_lang, self.pairs = self.get_word_info_and_pairs()
        if self.shuffle:
            random.shuffle(self.pairs)
        self.data_iter = iter(self.pairs)

    def get_word_info_and_pairs(self) -> Tuple[WordInfo, WordInfo, List[Tuple[str, str]]]:
        """  获取统计字数的模型和句子对 """
        # Read the file and split into lines
        pairs = []
        with open(self.data_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.strip().split("\t")
                if len(ss) == 2:
                    pairs.append([normalizeString(s) for s in ss])
        # filter pairs
        pairs = [pair for pair in pairs if
                 len(pair[0].split(' ')) < self.max_length and len(pair[1].split(' ')) < self.max_length]
        # word info
        input_lang = WordInfo(self.src_lang_name)
        output_lang = WordInfo(self.tar_lang_name)
        # add sentence
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        return input_lang, output_lang, pairs

    def __get_batch_pairs(self):
        """ 获取一批数据 """
        batch_pairs = []
        for pair in self.data_iter:
            batch_pairs.append(pair)
            if len(batch_pairs) >= self.batch_size:
                return batch_pairs

        return []

    def get_batch_tensor_data(self) -> Union[Tuple[torch.LongTensor, torch.LongTensor], None]:
        """ here, assuming that batch size is 1 """
        batch_pairs = self.__get_batch_pairs()
        if len(batch_pairs) < 1:
            return None
        for src_sen, tar_sen in batch_pairs:
            input_tensor = torch.LongTensor(
                [self.input_lang.word2index[word] for word in src_sen.split(" ")] + [self.eos_token])
            output_tensor = torch.LongTensor(
                [self.output_lang.word2index[word] for word in tar_sen.split(" ")] + [self.eos_token])
        return input_tensor, output_tensor

    def __iter__(self):
        return self

    def __next__(self):
        tensor_data = self.get_batch_tensor_data()
        if tensor_data is None:
            if self.shuffle:
                random.shuffle(self.pairs)
                self.data_iter = iter(self.pairs)
            raise StopIteration
        else:
            return tensor_data


if __name__ == "__main__":
    data = TranslationDataSet(data_path="data/eng-fra.txt")
    for epoch in range(50):
        for step, batch_data in enumerate(data):
            if step % 10000 == 0:
                print(epoch, step)
