'''
* @name: FeaDACF.py
* @description: Implementation of FeaDACF
'''

import torch
from torch import nn
from .bert import BertTextEncoder
from .MACF_Layers import *
from lstmforward import lstmforward



class MyNet(nn.Module):
    def __init__(self, dataset,bert_pretrained='./bert-base-uncased'):
        super(MyNet, self).__init__()

        self.max_feature_layers = 2
        self.h_hyper = nn.Parameter(torch.ones(1, 50, 128))
        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=bert_pretrained)

        if dataset == 'mosi':
            self.proj_l0 = nn.ModuleList()
            self.proj_l0.append(nn.LSTM(input_size=768, hidden_size=128, num_layers=1, batch_first=True))

            self.proj_a0 = nn.ModuleList()
            self.proj_a0.append(nn.LSTM(input_size=5, hidden_size=64, num_layers=1, batch_first=True))
            self.proj_a0.append(nn.Linear(64,128))

            self.proj_v0 = nn.ModuleList()
            self.proj_v0.append(nn.LSTM(input_size=20, hidden_size=32, num_layers=1, batch_first=True))
            self.proj_v0.append(nn.Linear(32,128))


        elif dataset == 'mosei':
            self.proj_l0 = nn.ModuleList()
            self.proj_l0.append(nn.LSTM(input_size=768, hidden_size=128, num_layers=1, batch_first=True))

            self.proj_a0 = nn.ModuleList()
            self.proj_a0.append(nn.LSTM(input_size=74, hidden_size=64, num_layers=1, batch_first=True))
            self.proj_a0.append(nn.Linear(64, 128))

            self.proj_v0 = nn.ModuleList()
            self.proj_v0.append(nn.LSTM(input_size=35, hidden_size=32, num_layers=1, batch_first=True))
            self.proj_v0.append(nn.Linear(32, 128))


        else:
            assert False, "DatasetName must be mosi, mosei ."


        #定义共享子空间编码器
        self.shared_encoder = SharedEncoder(input_size=128,shared_size=128)
        #定义特异性子空间编码器
        self.text_private_encoder = nn.Sequential(nn.Linear(128,128),
                                                  nn.ReLU())
        self.visual_private_encoder = nn.Sequential(nn.Linear(128,128),
                                                    nn.ReLU())
        self.audio_private_encoder = nn.Sequential(nn.Linear(128, 128),
                                                   nn.ReLU())

        #定义超模态协同跨模态注意力模块
        self.HSAM = HSAM2(dim=128,heads=4,dim_head=64,depth=4)
        #定义模态融合模块（融合同一个模态的两个子空间表征）
        self.ModalityFuse = ModalityFuse(input_size1=128,input_size2=128,output_size=128)
        
        self.t1 =  CrossTransformerBasedOnText(source_num_frames=50, tgt_num_frames=50, dim=128, depth=1, heads=8,
                                            dim_head=4, mlp_dim=128)
        self.t2 =  CrossTransformerBasedOnText(source_num_frames=50, tgt_num_frames=50, dim=128, depth=1, heads=8,
                                            dim_head=4, mlp_dim=128)
        

        self.ModalityCombination = ModalityCombination(input_dim1=128,input_dim2=128,output_dim=128)

        self.TRM = TransformerEncoder(dim = 896,depth=2,heads = 8,dim_head=128,mlp_dim=896)

        self.MLP = MultiLayerPredictor(input_size=896,hidden_size=512,output_size=1)

    def forward(self, x_visual, x_audio, x_text):
        # print(x_text.shape)  64,3,50

        x_text = self.bertmodel(x_text)

        # 此时真正意义上获得了三个模态的表征  shape为 b seq_length dm   但是必须注意此时它们的dim不一致 因此下面的代码将它们映射到了相同的dim

        r"""
        对于a，v两个模态，首先经过lstm 随后经过一个线性映射层，对于t模态直接使用线性映射层或LSTM其一即可.
        """
        self.x_visual = lstmforward(self.proj_v0,x_visual)
        self.x_audio = lstmforward(self.proj_a0, x_audio)
        self.x_text = lstmforward(self.proj_l0, x_text)
        r"""
        shape ； [batch  50  128]
        """
        r"""
        模态私有与共享表示学习模块
        首先将三种模态经过模态共享编码器映射到共享子空间。
        """

        self.x_visual_shared = self.shared_encoder(self.x_visual)
        self.x_audio_shared = self.shared_encoder(self.x_audio)
        self.x_text_shared = self.shared_encoder(self.x_text)

        self.x_visual_private = self.visual_private_encoder(self.x_visual)
        self.x_audio_private = self.audio_private_encoder(self.x_audio)
        self.x_text_private = self.text_private_encoder(self.x_text)

        r"""
        协同跨模态注意力
        将三个共享空间的模态表征输入到协同跨模态注意力机制层中.得到h_t与h_a
        """

        self.hyper,self.h_text= self.HSAM(self.x_text_shared,self.x_visual_shared,self.x_audio_shared)

        r"""
        双模态表示生成模块,
        
        首先将每个模态的私有空间和共享空间模态表征结合，得到统一的形式。
        """
     
        self.u_a = self.ModalityFuse(self.x_audio_shared,self.x_audio_private)
        self.u_v = self.ModalityFuse(self.x_visual_shared,self.x_visual_private)
        self.u_t = self.ModalityFuse(self.x_text_shared,self.x_text_private)

        r"""
        随后两两组合它们,得到双模态联合表示
        """
        # b_av = self.ModalityCombination(u_a,u_v)   we don't need this
        b_at = self.t1(self.u_a,self.u_t)
        b_vt = self.t2(self.u_t,self.u_v)

        r"""
        将上面得到的各种表征进行拼接
        """
        x = torch.cat((self.u_a,self.u_t,self.u_v,b_at,b_vt,self.hyper,self.h_text),dim=-1)

        feat = self.TRM(x)[:, 0]

        output = self.MLP(feat)

        return output


def build_model(opt):
    l_pretrained = 'bert-base-uncased'

    model = MyNet(dataset=opt.datasetName,bert_pretrained=l_pretrained)

    return model
