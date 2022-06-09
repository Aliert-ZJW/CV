'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import mobilenetV3_one as model_name
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

# Models
# default_model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))
#
# customized_models_names = sorted(name for name in customized_models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(customized_models.__dict__[name]))
#
# for name in customized_models.__dict__:
#     if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
#         models.__dict__[name] = customized_models.__dict__[name]
#
# model_names = default_model_names + customized_models_names


# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='/data/zjw/data/', type=str)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=1024, type=int, metavar='N', help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N', help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,  metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float, metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
# Checkpoint   一个文本文件，记录了训练过程中在所有中间节点上保存的模型的名称，首行记录的是最后（最近）一次保存的模型名称。
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Random seed  随机种子应该设置为多少?
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)
# if use_cuda:
#     torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):  # 判断checkpoint是不是目录
        mkdir_p(args.checkpoint)  # 不是目录创建目录 os.makedirs()

    # Data loading code
    traindir = os.path.join(args.data, 'train')  #os.path.join函数是实现路径拼接
    # valdir = os.path.join(args.data, 'val')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # 将数值变成[-1, 1]之间

     # 将数据集分成多个小组，每次抛出一组数据，直到把所有数据都抛出
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True, # 是非打乱顺序
        num_workers=args.workers, pin_memory=True)  #有多少个子进程用于数据加载
    print(len(train_loader), '=====================')
    # print('ss', train_loader)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print(val_loader)

    # print(train_loader)
    # for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
    #     print(inputs.shape, targets)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,4,5'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.device_count(), 'GPU')
    model = model_name.MobileNetV3_Large().to(device)
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model = model.cuda()

    # load_model = True
    # if (load_model):
    #     print('Warning! Loading pre-trained weights.')
    #     model.load_state_dict(torch.load('/data/zjw_plus/code/pretrained/mobilenetv3-large-1cd25616.pth'))

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'ImageNet-' + 'MobileNetV3'

    if args.resume:
        print('=============', args.resume)
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, True)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, True)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, True)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        print('val:({batch}/{size}) \tData: {data:.3f}s \tBatch: {bt:.3f}s  \tLoss: {loss:.4f} \ttop1: {top1: .4f} \ttop5: {top5: .4f}'.format(
            batch=batch_idx+1, size=len(val_loader),
            data=data_time.val,
            bt=batch_time.val,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        ))
    #     bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
    #         batch=batch_idx + 1,
    #         size=len(val_loader),
    #         data=data_time.avg,
    #         bt=batch_time.avg,
    #         total=bar.elapsed_td,
    #         eta=bar.eta_td,
    #         loss=losses.avg,
    #         top1=top1.avg,
    #         top5=top5.avg,
    #     )
    #     bar.next()
    # bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))

# def adjust_learning_rate(optimizer, epoch):
#     global state
#     if epoch in args.schedule:
#         state['lr'] *= args.gamma
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = state['lr']
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print('============change lr value', param_group['lr'])

def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            # inputs, targets = inputs.cuda(), targets.cuda(async=True)
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot progress
        print('train:({batch}/{size}) \tData: {data:.3f}s \tBatch: {bt:.3f}s  \tLoss: {loss:.4f} \ttop1: {top1: .4f} \ttop5: {top5: .4f}'.format(
            batch=batch_idx+1, size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        ))
    return (losses.avg, top1.avg)


if __name__ == '__main__':
    main()








