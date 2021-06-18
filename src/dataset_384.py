from collections import defaultdict
import random
import pandas as pd
import cv2

import torch
from torch.utils.data import Sampler
from torch.utils.data.dataset import Dataset

from bms import *
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose, Blur, RandomRotate90, RandomScale, CoarseDropout
    )


STOI = {
    '<sos>': 190,
    '<eos>': 191,
    '<pad>': 192,
}

image_size = 384
vocab_size = 193
max_length = 280


data_dir = '../input'


def read_pickle_from_file(pickle_file):
    with open(pickle_file, 'rb') as f:
        x = pickle.load(f)
    return x


def write_pickle_to_file(pickle_file, x):
    with open(pickle_file, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)


def pad_sequence_to_max_length(sequence, max_length, padding_value):
    batch_size =len(sequence)
    pad_sequence = np.full((batch_size, max_length), padding_value, np.int32)
    for b, s in enumerate(sequence):
        L = len(s)
        pad_sequence[b, :L, ...] = s
    return pad_sequence


def load_tokenizer():
    tokenizer = YNakamaTokenizer(is_load=True)
    print('len(tokenizer) : vocab_size', len(tokenizer))
    for k, v in STOI.items():
        assert tokenizer.stoi[k]==v
    return tokenizer


#(2424186, 6)
#Index(['image_id', 'InChI', 'formula', 'text', 'sequence', 'length'], dtype='object')
def make_fold(mode='train-1'):
    if 'train' in mode:
        df = read_pickle_from_file(data_dir+'/df_train.more.csv.pickle')
        #df = pd.read_csv(data_dir + 'train_labels.csv')
        df_fold = pd.read_csv(data_dir+'/df_fold.csv')
        df = df.merge(df_fold, on='image_id')
        df.loc[:, 'path'] = 'train'
        df.loc[:, 'orientation'] = 0

        df['fold'] = df['fold'].astype(int)
        #print(df.groupby(['fold']).size()) #404_031
        #print(df.columns)

        fold = int(mode[-1])
        df_train = df[df.fold != fold].reset_index(drop=True)
        df_valid = df[df.fold == fold].reset_index(drop=True)
        return df_train, df_valid

    # Index(['image_id', 'InChI'], dtype='object')
    if 'test' in mode:
        #df = pd.read_csv(data_dir+'/sample_submission.csv')
        df = pd.read_csv(data_dir+'/submit_lb0.65.csv')
        df_orientation = pd.read_csv(data_dir+'/test_orientation.csv')
        df = df.merge(df_orientation, on='image_id')

        df.loc[:, 'path'] = 'test'
        #df.loc[:, 'InChI'] = '0'
        df.loc[:, 'formula'] = '0'
        df.loc[:, 'text'] =  '0'
        df.loc[:, 'sequence'] = pd.Series([[0]] * len(df))
        df.loc[:, 'length'] = df.InChI.str.len()

        df_test = df
        return df_test


#####################################################################################################
class FixNumSampler(Sampler):
    def __init__(self, dataset, length=-1, is_shuffle=False):
        if length <= 0:
            length = len(dataset)

        self.is_shuffle = is_shuffle
        self.length = length

    def __iter__(self):
        index = np.arange(self.length)
        if self.is_shuffle:
            random.shuffle(index)
        return iter(index)

    def __len__(self):
        return self.length


# see https://www.kaggle.com/yasufuminakama/inchi-resnet-lstm-with-attention-inference/data
def remote_unrotate_augment(r):
    image = r['image']
    h, w = image.shape

    if h > w:
         image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #l = r['d'].orientation
    #if l == 1:
    #    image = np.rot90(image, -1)
    #if l == 2:
    #    image = np.rot90(image, 1)
    #if l == 3:
    #    image = np.rot90(image, 2)

    image = cv2.resize(image, dsize=(image_size,image_size), interpolation=cv2.INTER_LINEAR)
    assert image_size == 384

    r['image'] = image
    return r


def null_augment(r):
    image = r['image']
    image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)
    assert image_size==384
    r['image'] = image
    return r


def get_augmentation():
    transform = [
        #RandomRotate90(p=0.01),
        RandomScale(scale_limit=(-0.2, +0.2), interpolation=1, always_apply=False, p=0.3),
        Cutout(num_holes=100, max_h_size=1, max_w_size=1, always_apply=False, p=0.3),
        #RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.0, brightness_by_max=True, always_apply=False, p=0.3),
        #CoarseDropout(max_holes=150, max_height=1, max_width=1,
        #              min_holes=50, min_height=1, min_width=1,
        #              fill_value=0, mask_fill_value=None, always_apply=False, p=0.3),
    ]
    return Compose(transform)


def null_augment_tr(r):

    image = r['image']
    trans = get_augmentation()
    image = trans(image=image)['image']

    image = cv2.resize(image, dsize=(image_size,image_size), interpolation=cv2.INTER_LINEAR)
    assert image_size == 384

    r['image'] = image
    return r


class BmsDataset(Dataset):
    def __init__(self, df, tokenizer, augment=null_augment):
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df
        self.augment = augment
        self.length = len(self.df)

    def __str__(self):
        string = ''
        string += '\tlen = %d\n'%len(self)
        string += '\tdf  = %s\n'%str(self.df.shape)
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]

        image_file = data_dir +'/%s/%s/%s/%s/%s.png'%(d.path, d.image_id[0], d.image_id[1], d.image_id[2], d.image_id)
        #image_file = data_dir +'/train/%s/%s/%s/%s.png'%(d.image_id[0], d.image_id[1], d.image_id[2], d.image_id)
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        token = d.sequence
        r = {
            'index': index,
            'image_id': d.image_id,
            'InChI': d.InChI,
            'formula': d.formula,
            'd': d,
            'image': image,
            'token': token,
        }
        if self.augment is not None:
            r = self.augment(r)
        return r


def null_collate(batch, is_sort_decreasing_length=True):
    collate = defaultdict(list)

    if is_sort_decreasing_length:  # sort by decreasing length
        sort = np.argsort([-len(r['token']) for r in batch])
        batch = [batch[s] for s in sort]

    for r in batch:
        for k, v in r.items():
            collate[k].append(v)

    collate['length'] = [len(l) for l in collate['token']]

    token = [np.array(t, np.int32) for t in collate['token']]
    token = pad_sequence_to_max_length(token, max_length=max_length, padding_value=STOI['<pad>'])
    collate['token'] = torch.from_numpy(token).long()

    image = np.stack(collate['image'])
    image = image.astype(np.float32) / 255
    collate['image'] = torch.from_numpy(image).unsqueeze(1).repeat(1, 3, 1, 1)

    return collate
