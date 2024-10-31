'''
* @name: opts.py
* @description: Hyperparameter configuration. Note: For hyperparameter settings, please refer to the appendix of the paper.
'''


import argparse

def parse_opts():
    parser = argparse.ArgumentParser()   #首先获取一个ArgumentParser解析器对象
    arguments = {  #创建一个命令行字典
        'dataset': [
            dict(name='--datasetName',        
                 type=str,
                 default='mosi',
                 help='mosi, mosei or sims'),
            dict(name='--dataPath',
                 default="./datasets/unaligned_50.pkl",
                 type=str,
                 help=' '),
            dict(name='--seq_lens',     
                 default=[50, 50, 50],
                 type=list,
                 help=' '),
            dict(name='--num_workers',
                 default=8,
                 type=int,
                 help=' '),
           dict(name='--train_mode',
                 default="regression",
                 type=str,
                 help=' '),
            dict(name='--test_checkpoint',
                 default="./checkpoint/test/SIMS_Acc7_Best.pth",
                 type=str,
                 help=' '),
        ],
        'network': [
            dict(name='--CUDA_VISIBLE_DEVICES',        
                 default='6',
                 type=str),
            dict(name='--fusion_layer_depth',
                 default=2,
                 type=int)
        ],

        'common': [
            dict(name='--project_name',    
                 default='ALMT_Demo',
                 type=str
                 ),
           dict(name='--is_test',    
                 default=1,
                 type=int
                 ),
            dict(name='--seed',  # try different seeds
                 default=18,
                 type=int
                 ),
            dict(name='--models_save_root',
                 default='./checkpoint',
                 type=str
                 ),
            dict(name='--batch_size',
                 default=64,
                 type=int,
                 help=' '),
            dict(
                name='--n_threads',
                default=3,
                type=int,
                help='Number of threads for multi-thread loading',
            ),
            dict(name='--lr',
                 type=float,
                 default=1e-4),
            dict(name='--weight_decay',
                 type=float,
                 default=1e-4),
            dict(
                name='--n_epochs',
                default=10,
                type=int,
                help='Number of total epochs to run',
            )
        ]
    }

    for group in arguments.values():  #遍历字典arguments，每一个group都是这个字典的一个键所对应的值（一个元素为字典的列表）
        for argument in group:   #每一个argument都是一个字典
            name = argument['name'] #取出字典中键位name所对应的值
            del argument['name']  #删除字典argument中的name键值对
            # **kwargs允许你在函数中处理那些关键字参数，它们在函数被调用时未被明确指定。
            # 因此，字典的键名词必须严格规范default不能写成Default或其他任何一种
            # 这些参数被打包进一个字典中，使得函数能够以更灵活的方式接收数据
            parser.add_argument(name, **argument)
            # 注意 我们在运行程序的时候 如果使用terminal ，那么在终端利用输入--datasetName hhh 可以把datasetName原有的默认值更改为hhh
            # parser会非常聪明地解析你所输入的参数
    args = parser.parse_args() # 解析参数
    return args  #args是一个命名空间namespace对象  可以用args.name取出键name所对应的值