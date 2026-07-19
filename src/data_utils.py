import re
import json
import torch

SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>", 3: "<UNK>"}
        self.n_words = 4  # Count SOS, EOS, PAD, UNK

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

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()

def read_faq_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pairs = []
    for item in data:
        q = normalizeString(item['question'])
        a = normalizeString(item['answer'])
        pairs.append([q, a])
    return pairs

def prepareData(filepath):
    pairs = read_faq_data(filepath)
    vocab = Lang("vocab")
    for pair in pairs:
        vocab.addSentence(pair[0])
        vocab.addSentence(pair[1])
    return vocab, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index.get(word, UNK_token) for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence, max_length):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    # Padding
    while len(indexes) < max_length:
        indexes.append(PAD_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

def tensorsFromPair(pair, lang, max_length=15):
    input_tensor = tensorFromSentence(lang, pair[0], max_length)
    target_tensor = tensorFromSentence(lang, pair[1], max_length)
    return (input_tensor, target_tensor)
