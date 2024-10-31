import os
import sys

import torch
import numpy as np
from tqdm import tqdm
from opts import *
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, save_model, setup_seed
from tensorboardX import SummaryWriter
from models.FeaDAM2 import build_model
from core.metric import MetricsTop
from utils.functions import DiffLoss, CMD
from thop import profile
from sklearn import manifold
# 命令行与显卡的准备工作
opt = parse_opts()  # 调用编写好的该函数将返回一个namespace，namespace类似于字典，可以用opt.name来获取值
# 在cmd中输入 nvidia-smi可以看到显卡编号，一般情况下，0号为主卡
# 例如主机有四个卡，则os.environ["CUDA_VISIBLE_DEVICES"] = ‘3，2，1，0’将指定3号卡为运算显卡编号为0的主卡。
# 但是实际显卡编号依然是不变的
os.environ["CUDA_VISIBLE_DEVICES"] = opt.CUDA_VISIBLE_DEVICES
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
print("device: {}:{}".format(device, opt.CUDA_VISIBLE_DEVICES))

# 用于可视化训练过程
train_mae, val_mae = [], []


def main():
    torch.cuda.empty_cache()
    opt = parse_opts()
    # 设置随机数种子方便复现实验
    if opt.seed is not None:
        setup_seed(opt.seed)
    print("seed: {}".format(opt.seed))
    # 设置项目路径
    log_path = os.path.join(".", "log", opt.project_name)
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    print("log_path :", log_path)  # .\log\project_name
    # 设置保存路径
    save_path = os.path.join(opt.models_save_root, opt.project_name)
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    print("model_save_path :", save_path)

    # 创建模型
    model = build_model(opt).to(device)
    # 创建采样器
    dataLoader = MMDataLoader(opt)
    # 设置优化算法
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=opt.lr,
                                  weight_decay=opt.weight_decay)

    scheduler_warmup = get_scheduler(optimizer, opt)
    # 计算损失函数
    loss_fn = torch.nn.MSELoss()  # MSE误差 这里属于的是Ltask损失
    ldiff = DiffLoss()

    metrics = MetricsTop().getMetics(opt.datasetName)

    writer = SummaryWriter(logdir=log_path)
    i = 0

    for epoch in range(1, opt.n_epochs + 1):
        i = i + 1
        train(model, dataLoader['train'], optimizer, loss_fn, ldiff, epoch, writer, metrics)
        evaluate(model, dataLoader['valid'], optimizer, loss_fn, ldiff, epoch, writer, save_path, metrics,i)
        if opt.is_test is not None:
            test(model, dataLoader['test'], optimizer, loss_fn, ldiff, epoch, writer, metrics)
        scheduler_warmup.step()
    writer.close()


def train(model, train_loader, optimizer, loss_fn, ldiff, epoch, writer, metrics):
    train_pbar = tqdm(enumerate(train_loader))
    losses = AverageMeter()

    y_pred, y_true = [], []

    model.train()
    for cur_iter, data in train_pbar:
        img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)

        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        # 模型前向传播


        output = model(img, audio, text)

        loss_df = ldiff(model.x_audio_private, model.x_audio_shared)
        loss_df += ldiff(model.x_visual_private, model.x_visual_shared)
        loss_df += ldiff(model.x_text_private, model.x_text_shared)
        loss_df += ldiff(model.x_audio_private, model.x_text_private)
        loss_df += ldiff(model.x_audio_private, model.x_visual_private)
        loss_df += ldiff(model.x_text_private, model.x_visual_private)

        loss = loss_fn(output, label) + loss_df * 0.3

        losses.update(loss.item(), batchsize)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

        train_pbar.set_description('train')
        train_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                'loss': '{:.5f}'.format(losses.value_avg),
                                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)
    print('train: ', train_results)
    train_mae.append(train_results['MAE'])

    writer.add_scalar('train/loss', losses.value_avg, epoch)


def evaluate(model, eval_loader, optimizer, loss_fn, ldiff, epoch, writer, save_path, metrics,count):
    test_pbar = tqdm(enumerate(eval_loader))

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            output = model(img, audio, text)
            loss_df = ldiff(model.x_audio_private, model.x_audio_shared)
            loss_df += ldiff(model.x_visual_private, model.x_visual_shared)
            loss_df += ldiff(model.x_text_private, model.x_text_shared)
            loss_df += ldiff(model.x_audio_private, model.x_text_private)
            loss_df += ldiff(model.x_audio_private, model.x_visual_private)
            loss_df += ldiff(model.x_text_private, model.x_visual_private)

            loss = loss_fn(output, label) + loss_df * 0.3

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

            test_pbar.set_description('eval')
            test_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                   'loss': '{:.5f}'.format(losses.value_avg),
                                   'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        print(test_results)

        if (count % 4 == 0):
          writer.add_scalar('evaluate/loss', losses.value_avg, epoch)

          save_model(save_path, epoch, model, optimizer)


def test(model, test_loader, optimizer, loss_fn, ldiff, epoch, writer, metrics):
    test_pbar = tqdm(enumerate(test_loader))

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            output = model(img, audio, text)
            loss_df = ldiff(model.x_audio_private, model.x_audio_shared)
            loss_df += ldiff(model.x_visual_private, model.x_visual_shared)
            loss_df += ldiff(model.x_text_private, model.x_text_shared)
            loss_df += ldiff(model.x_audio_private, model.x_text_private)
            loss_df += ldiff(model.x_audio_private, model.x_visual_private)
            loss_df += ldiff(model.x_text_private, model.x_visual_private)

            loss = loss_fn(output, label) + loss_df * 0.3

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

            test_pbar.set_description('test')
            test_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                   'loss': '{:.5f}'.format(losses.value_avg),
                                   'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        print(test_results)

        writer.add_scalar('test/loss', losses.value_avg, epoch)


if __name__ == '__main__':
    main()



