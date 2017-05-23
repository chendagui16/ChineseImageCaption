# @ Author: Dagui Chen
# @ Email: goblin_chen@163.com
# @ Date: 2017-05-02
# ========================================
import numpy as np
import h5py


class Preprocessor(object):
    """
    According to the config, preprocess the data to get the feature veator
    and its corresponding caption
    """
    def __init__(self, config):
        f = h5py.File(config.feature_path, 'r')
        self.train_set = f['train_set'].value
	if config.ifpool:
            self.train_num = self.train_set.shape[0]
	    self.image_len = self.train_set.shape[-1]
#	self.train_num, self.image_len = self.train_set.shape
	else:
	    self.train_num, self.image_len = self.train_set.shape

        self.val_set = f['validation_set'].value
        self.val_num = self.val_set.shape[0]

        self.test_set = f['test_set'].value
        self.test_num = self.test_set.shape[0]

        f.close()
        self.train_captions = self.parse_caption(config.caption, 'train')
        self.val_captions = self.parse_caption(config.caption, 'valid')

        self.caption_len = 0
        for caption in self.train_captions + self.val_captions:
            maxlen = max(map(len, (x for x in caption)))
            if maxlen > self.caption_len:
                self.caption_len = maxlen

    def parse_caption(self, caption_file, dataset='train'):
        assert dataset in ['train', 'valid']
        f = open(caption_file.format(dataset), 'r')
        lines = f.readlines()
        f.close()
        lines[0] = u'1'
        captions = []
        caption = []
        for line in lines:
            line = line.decode('utf-8').strip()
            if line.isdigit() == True:
                if(caption != []):
                    captions.append(caption)
                caption = []
            else:
                caption.append(self.tokenize(line))
        captions.append(caption)
        return captions

    def tokenize(self, sent):
        return [x.strip() for x in sent]


class CharaterTable(object):
    """Given a set of captions:
    + Encode them to a one-hot integer representaion
    + Decode the one-hot integer representation to their character output
    + Decode the integer representation to charater
    """
    def __init__(self, captions):
        self.start_word = '<w>'
        self.stop_word = '<\w>'
        vocabs = set((self.start_word, self.stop_word))
        for caption in captions:
            for line in caption:
                vocabs |= set(line)
        vocabs = sorted(vocabs)
        self.vocab_size = len(vocabs)
        self.char_indices = dict((c, i) for i, c in enumerate(vocabs))
        self.indices_char = dict((i, c) for i, c in enumerate(vocabs))
        self.start_idx = self.char_indices[self.start_word]
        self.stop_idx = self.char_indices[self.stop_word]

    def encode(self, C, num_rows):
        x = np.zeros((num_rows + 2, self.vocab_size))
        x[0, self.start_idx] = 1
        for i, c in enumerate(C):
            if i >= num_rows:  # restrict the caption length
                break
            x[i+1, self.char_indices[c]] = 1
        x[i+2:, self.stop_idx] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        words = []
        find_start = False
        for i in x:
            if i == self.start_idx:
                find_start = True
                continue
            elif i == self.stop_idx:
                break
            if find_start:
                words.append(self.indices_char[i])
        return words


def vetorize_caption(captions, ctable, caption_len):
    Y = []
    for caption in captions:
        subY = []
        for line in caption:
            subY.append(ctable.encode(line, caption_len))
        Y.append(subY)
    return Y
