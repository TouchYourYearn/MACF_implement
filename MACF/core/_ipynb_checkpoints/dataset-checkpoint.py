'''
* @name: dataset.py
* @description: Dataset loading functions. Note: The code source references MMSA (https://github.com/thuiar/MMSA/tree/master).
'''


import logging
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims
        }
        DATA_MAP[args.datasetName]() #python字典变量的值可以是一个函数，然后也可以利用 字典名称[键的名字]（）  来实现调用

    def __init_mosi(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)
        # pkl文件是python里面保存文件的一种格式，如果直接打开会显示一堆序列化的东西（二进制文件）。
        #  常用于保存神经网络训练的模型或者各种需要存储的数据。
        #  上面的代码从pickle格式的文件中读取数据并转换为Python的类型。
        r"""
        print(len(data['train']['raw_text']))  #1284
        print(len(data['train']['audio']))   #1284
        print(len(data['train']['vision']))  #1284 375 5
        print(len(data['train']['text'])) #1284
        print(len(data['train']['audio_lengths'])) #1284
        print(len(data['train']['vision_lengths'])) #1284
        print(data['train']['audio_lengths'])  #1284个视频，每个视频的长度组成的列表
        print(data['train']['audio'][1].shape)   #375 5  很多0
        print(data['train']['vision'][1].shape)  #500, 20  很多0
        print(data['train']['text_bert'][1])  # 3,50  3指的是词ID，句子id，和掩码id的值
        print(data['train']['audio'][1])
        print(data['train']['vision'][1])

        non_zero_rows = np.count_nonzero(np.count_nonzero(data['train']['audio'][1], axis=1))  #17
        print(data['train']['visions_lengths'][1])   #17
        print(non_zero_rows)

        """
        #print(data['train']['raw_text 这个就是原始的文本序列
        #print(data['train']['text_bert']) 
        print(data['train']['text_bert'][0])  
        print(data['train']['text_bert'][12]) 

        #data['train']['raw_text'] = np.array(data['train']['raw_text'])  # 将数据转换为np数组
        # print(len(data['train']['raw_text']))
        #print(data['train']['vision_lengths'][1])  # 17
        # print(data['train']['raw_text'])
        # sys.exit()
        #
        # print(data['train']['audio'][0:5,:,:])
        # sys.exit()
        #  # print(data['train']['audio'].dtype)
        #  print(data['train']['regression_labels'].shape)
        # data['train']['index'] = torch.tensor(range(1284))   #记录数据的索引 0 1 2 3 ..... 1284
        #  print(data['train']['index'])
        # sys.exit()


        self.args.use_bert = True
        self.args.need_truncated = True
        self.args.need_data_aligned = True

        if self.args.use_bert:
            self.text = data[self.mode]['text_bert'].astype(np.float32)

        else:
            self.text = data[self.mode]['text'].astype(np.float32)
     
        self.vision = data[self.mode]['vision'].astype(np.float32)

        self.audio = data[self.mode]['audio'].astype(np.float32)

        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']
        self.labels = {
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32)
        }
        if self.args.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m]

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args.need_data_aligned:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        if self.args.need_truncated:
            self.__truncated()

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __truncated(self):
        # NOTE: Here for dataset we manually cut the input into specific length.
        def Truncated(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):    #沿着序列长度遍历
                    if((instance[index] == padding).all()): #如果遍历的过程中发现某一行全为0
                        if(index + length >= modal_features.shape[1]):   #如果第一个全为0的行索引值 + length > 总长度
                            truncated_feature.append(instance[index:index+length])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+length])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature

        audio_length, video_length =[50,50] 

        self.vision = Truncated(self.vision, video_length)
        self.audio = Truncated(self.audio, audio_length)

        

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.use_bert:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.rawText[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        if not self.args.need_data_aligned:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        return sample


def MMDataLoader(args):   #实践中这里传入的是opt namespace
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len()    #论文与默认情况下为[50,50,50]

    dataLoader = {  #这里同时创建了三个采样器分别为train valid test
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader