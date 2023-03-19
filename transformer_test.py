import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np


from download import download
from pathlib import Path
from tqdm import tqdm
import os

# 训练、验证、测试数据集下载地址
urls = {
    'train': 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz',
    'valid': 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz',
    'test': 'http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_test2016.tar.gz'
}

# 指定保存路径为 `home_path/.mindspore_examples`
cache_dir = Path.home() / '.mindspore_examples'

train_path = download(urls['train'], os.path.join(cache_dir, 'train'), kind='tar.gz')
valid_path = download(urls['valid'], os.path.join(cache_dir, 'valid'), kind='tar.gz')
test_path = download(urls['test'], os.path.join(cache_dir, 'test'), kind='tar.gz')

import spacy
from functools import partial

class Multi30K():
    """Multi30K数据集加载器

    加载Multi30K数据集并处理为一个Python迭代对象。

    """
    def __init__(self, path):
        self.data = self._load(path)

    def _load(self, path):
        def tokenize(text, spacy_lang):
            # 去除多余空格，统一大小写
            text = text.rstrip()
            return [tok.text.lower() for tok in spacy_lang.tokenizer(text)]

        # 加载英、德语分词器
        tokenize_de = partial(tokenize, spacy_lang=spacy.load('de_core_news_sm'))
        tokenize_en = partial(tokenize, spacy_lang=spacy.load('en_core_web_sm'))

        # 读取Multi30K数据，并进行分词
        members = {i.split('.')[-1]: i for i in os.listdir(path)}
        de_path = os.path.join(path, members['de'])
        en_path = os.path.join(path, members['en'])
        with open(de_path, 'r', encoding='utf-8') as de_file:
            de = de_file.readlines()[:-1]
            de = [tokenize_de(i) for i in de]
        with open(en_path, 'r', encoding='utf-8') as en_file:
            en = en_file.readlines()[:-1]
            en = [tokenize_en(i) for i in en]

        return list(zip(de, en))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

train_dataset, valid_dataset, test_dataset = Multi30K(train_path), Multi30K(valid_path), Multi30K(test_path)

class Vocab:
    """通过词频字典，构建词典"""

    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']

    def __init__(self, word_count_dict, min_freq=1):
        self.word2idx = {}
        for idx, tok in enumerate(self.special_tokens):
            self.word2idx[tok] = idx

        # 过滤低词频的词元
        filted_dict = {
            w: c
            for w, c in word_count_dict.items() if c >= min_freq
        }
        for w, _ in filted_dict.items():
            self.word2idx[w] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        self.bos_idx = self.word2idx['<bos>']  # 特殊占位符：序列开始
        self.eos_idx = self.word2idx['<eos>']  # 特殊占位符：序列结束
        self.pad_idx = self.word2idx['<pad>']  # 特殊占位符：补充字符
        self.unk_idx = self.word2idx['<unk>']  # 特殊占位符：低词频词元或未曾出现的词元

    def _word2idx(self, word):
        """单词映射至数字索引"""
        if word not in self.word2idx:
            return self.unk_idx
        return self.word2idx[word]

    def _idx2word(self, idx):
        """数字索引映射至单词"""
        if idx not in self.idx2word:
            raise ValueError('input index is not in vocabulary.')
        return self.idx2word[idx]

    def encode(self, word_or_list):
        """将单个单词或单词数组映射至单个数字索引或数字索引数组"""
        if isinstance(word_or_list, list):
            return [self._word2idx(i) for i in word_or_list]
        return self._word2idx(word_or_list)

    def decode(self, idx_or_list):
        """将单个数字索引或数字索引数组映射至单个单词或单词数组"""
        if isinstance(idx_or_list, list):
            return [self._idx2word(i) for i in idx_or_list]
        return self._idx2word(idx_or_list)

    def __len__(self):
        return len(self.word2idx)
    
from collections import Counter, OrderedDict

def build_vocab(dataset):
    de_words, en_words = [], []
    for de, en in dataset:
        de_words.extend(de)
        en_words.extend(en)

    de_count_dict = OrderedDict(sorted(Counter(de_words).items(), key=lambda t: t[1], reverse=True))
    en_count_dict = OrderedDict(sorted(Counter(en_words).items(), key=lambda t: t[1], reverse=True))

    return Vocab(de_count_dict, min_freq=2), Vocab(en_count_dict, min_freq=2)

de_vocab, en_vocab = build_vocab(train_dataset)
trg_pad_idx = en_vocab.pad_idx

net_work = nn.Transformer()
loss_fn = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
optimizer = nn.Adam(net_work.trainable_params(), learning_rate=0.0001)

src = Tensor(np.random.rand(128, 32, 512), mindspore.float32)
trg = Tensor(np.random.rand(128, 32, 512), mindspore.float32)

def forward_fn(src, trg):
    out = net_work(src, trg)
    loss = loss_fn(out, trg)
    return loss


grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters)

def train_step(src, trg):
    loss, grads = grad_fn(src, trg)
    optimizer(grads)
    return loss

net_work.set_train(True)   
train_step(src, trg)