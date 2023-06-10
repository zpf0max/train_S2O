from __future__ import print_function
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12138'
import torch
from models.modeling import VisionTransformer, CONFIGS
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy
from models.preact_resnet import *
from models.wideresnet import *
from trades_s2o import trades_loss
import torch.distributed as dist
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

os.environ['TORCH_HOME']=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'model/')
import time
import numpy as np
import timm
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import cv2

import third_party.efficientnet_pytorch as efficientnet_pytorch
from torchvision import transforms

from pytorch_ares.attack_torch import *
from timm.utils import AverageMeter
import torch.nn.functional as F
from PIL import Image
import time
import math
from tqdm import tqdm
import pdb

from networks import *
from torch.autograd import Variable
parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8/255,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2/255,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar10-wideresnet/adv_preactresnet18/',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=100, type=int, metavar='N',
                    help='save frequency')

parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
args = parser.parse_args(args=[]) 

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
torch.cuda.set_device(args.local_rank)
'''
torch.distributed.init_process_group(
    backend='nccl',
    init_method='env://',
    #timeout=datetime.timedelta(0, 1800),
    world_size=1,
    rank=0,
    store=None,
    group_name=''
)


'''

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
#train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
train_loader = torch.utils.data.DataLoader(trainset , batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
#test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
'''

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
train_loader = torch.utils.data.DataLoader(train_sampler , batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
'''

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    model.eval()
    out = model(X)

    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss2 = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss2.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd
def get_model(model):
    return deepcopy(model.state_dict())

def set_model(model,state_dict):
    model.load_state_dict(copy.deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def criterion(output,targets,fisher2):
        # Regularization for all previous tasks
    loss_reg=0
    for (name,param),(_,param_old) in zip(model.named_parameters(),model_old.named_parameters()):
        #print(loss_reg.is_cuda)
        loss_reg+=torch.sum(fisher2[name]*(param_old-param).pow(2))/2
    
    return ce(output,targets)+100000/args.batch_size*loss_reg
def fisher_matrix_diag(x,y,model,criterion,sbatch=20,epoch=0):
    # Init
    if epoch==0:
        for n,p in model.named_parameters():
            fisher[n]=0*p.data
    fisher_old=copy.deepcopy(fisher)
    
    # Compute
    #print(fisher)
    model.train()
    num=0
    
    
    for i in tqdm(range(0,x.size(0),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        loss=0
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
        #print(b)
        images=torch.autograd.Variable(x[b],volatile=False)
        target=torch.autograd.Variable(y[b],volatile=False)
        # Forward and backward
        model.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,target,fisher_old)
        loss.backward()
        # Get gradients
        num+=1
        for n,p in model.named_parameters():
            if p.grad is not None:   
                fisher[n]+=sbatch*p.grad.data.pow(2) 
    # Mean          return
    
    for n,p in model.named_parameters():
        fisher[n]=fisher[n]/x.size(0)
        fisher[n]=(fisher_old[n]*epoch+fisher[n])/(epoch+1)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher
def train(args,model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    #torch_resize =torchvision.transforms.Resize([224,224]) 
    #torch_resize2 =torchvision.transforms.Resize([32,32]) 
    #train_loader2=
    for param in model_old.parameters():
        param.requires_grad = True
    fisher_old=copy.deepcopy(fisher)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = data.to(device), target.to(device)
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        '''
        #print(data.shape)
        #data, target=Variable(data),Variable(target)
        #data=torch_resize(data)
        outputs=model(data)
        import pdb
        
        loss=criterion(outputs,target,fisher_old)*0.01

        #print(loss)
        #output=model(data)[0]
        #data=torch_resize2(data)
        #loss_adv =F.cross_entropy(output, target, size_average=True)
        
        with torch.no_grad():
            X, y = Variable(data, requires_grad=True), Variable(target)
            data_adv = _pgd_whitebox(copy.deepcopy(model), X, y)
        outputs2 = model(data_adv)
        loss2=criterion(outputs2,target,fisher_old)*0.99
        '''
        #print("loss",los
        #print("loss2",loss2)

        #loss_adv=loss+loss2
        #print(loss_adv)
        #loss2=ce(output0,target)*0.15
        loss_adv= trades_loss(model=model,  
                              model_old=model_old, 
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           fisher_old=fisher_old,
                           perturb_steps=args.num_steps,
                           beta=args.beta,
                           )
        
        #loss1 = loss_adv.detach_().requires_grad_(True)
        #for n,_ in model.named_parameters():
        #    fisher[n]=fisher[n]/50000
        #   fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
        loss_adv.backward()
        optimizer.step()
    for param in model_old.parameters():
        param.requires_grad = False
def train_adv0(args,model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    #torch_resize =torchvision.transforms.Resize([224,224]) 
    #torch_resize2 =torchvision.transforms.Resize([32,32]) 
    #train_loader2=
    for param in model_old.parameters():
        param.requires_grad = True
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = data.to(device), target.to(device)
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        
        loss_adv = trades_loss(model=model,
                            
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta,
                        )
        
        #loss1 = loss_adv.detach_().requires_grad_(True)
        #for n,_ in model.named_parameters():
        #    fisher[n]=fisher[n]/50000
        #   fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
        loss_adv.backward()
        optimizer.step()
   
def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    correct2= 0
    torch_resize =torchvision.transforms.Resize([224,224]) 
    with torch.no_grad():
        for data, target in test_loader:
            #data, target = data.cuda(), target.to(device)
            data, target = data.cuda(), target.cuda()
            
            with torch.no_grad():
                X, y = Variable(data, requires_grad=True), Variable(target)
                data_adv = _pgd_whitebox(copy.deepcopy(model), X, y)
            #data_adv=torch_resize(data_adv)
            #data_adv=data
            output = model(data_adv)
            output2=model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            pred2=output2.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct2 += pred2.eq(target.view_as(pred2)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    test_accuracy2 = correct2 / len(test_loader.dataset)
    return test_loss, test_accuracy,test_accuracy2

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.1
    if epoch >= 150:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_model_(model,state_dict):
    model.load_state_dict(copy.deepcopy(state_dict))
    return

def main():
    # init model, ResNet18() can be also used here for training
    num_classes=10
    global fisher
    fisher={}
    t=0
    global model
    global model_old
    global ce 
    global model_resnet50

    ce=torch.nn.CrossEntropyLoss()
    #model = WideResNet(num_classes=10)#.to(device)
    #model_old = WideResNet(num_classes=10)#.to(device)
    #model=ResNet(50, num_classes)
    #model_old=ResNet(50, num_classes)
    model=PreActResNet18(num_classes=10)
    model_old=PreActResNet18(num_classes=10)
    #model_resnet50=WideResNet(num_classes=10)#.to(device)
    sign=True
    #model0 = VisionTransformer(config, 224, zero_head=True, num_classes=num_classes)
    #model0.load_state_dict(torch.load("/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/output/cifar10-100_500_checkpoint.bin"))
    #model0.cuda()
    #pretrain_weights_path = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/model-cifar10-wideresnet/result_adv_train/wide_resnet50_EWC_10_no.pt"
    #pretrain_weights_path = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/checkpoint/cifar10/resnet-50finetune.t7"
    #pretrain_weights_path2 = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/checkpoint/cifar10/resnet-50origin-no.t7"
    
    #pretrain_weights_path = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/model-cifar10-wideresnet/0.4/epoch128.pt"
    #pretrain_weights_path = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/model-cifar10-wideresnet/result_adv_train/PREACTresnet18_10_no.pt"
    #pretrain_weights_path2 = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/model-cifar10-wideresnet/0.3/epoch103.pt" #wideresnet_adv_train
    '''
    #pretrain_weights_path2 = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/model-cifar10-wideresnet/result_adv_train/resnet50_EWC_10.pt"
    weights= torch.load(pretrain_weights_path)
    '''
    pretrain_weights_path="/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/model-cifar10-wideresnet/wide_resnet/epoch19.pt"
    weights_dict = {}
    weights= torch.load(pretrain_weights_path)
    for k, v in weights.items():
        #for k, v in weights['net']:
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] =v     
    model.load_state_dict(weights_dict)
    
    #model=weights['net']
    model=model.cuda()
    '''
    weights2= torch.load(pretrain_weights_path2)
    weights_dict2 = {}
    for k, v in weights2.items():
        #for k, v in weights['net']:
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict2[new_k] =v   
    model_resnet50.load_state_dict(weights_dict2)
    model_resnet50.cuda()
    '''
    #model = torch.nn.parallel.DistributedDataParallel(model,broadcast_buffers=False, find_unused_parameters=True)
    #model = torch.nn.DataParallel(model)
    #model_old = torch.nn.DataParallel(model)
    #model = PreActResNet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    data0=torch.tensor([])
    target0=torch.LongTensor([])
    data0, target0= data0.cuda(), target0.cuda()
    tstt = []
    train_loader2 = torch.utils.data.DataLoader(trainset , batch_size=100,shuffle=True, **kwargs)
    for i,(data,target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data0=torch.cat((data0,data.clone()),dim=0)  
        target0=torch.cat((target0,target.clone()),dim=0)
    patience=8
    tstloss, tst_acc,tst_acc_origin = eval_test(model, device, test_loader)
    print('test_loss: {:.4f}, test_acc: {:.2f}%,test_acc_origin: {:.2f}%'.format(tstloss, 100. * tst_acc,100.*tst_acc_origin))
    #fisher=fisher_matrix_diag(data0,target0,model,criterion,sbatch=200,epoch=0)
    #tstloss, tstacc,tst_acc_origin = eval_test(model, device, test_loader)
    #print('test_loss: {:.4f}, test_acc: {:.2f}%,test_acc_origin: {:.2f}%'.format(tstloss, 100. * tstacc,100.*tst_acc_origin))
    base=1
    model_old.load_state_dict(copy.deepcopy(model.state_dict()))
                    #model_old=model_old.cuda()
    freeze_model(model_old) # Freeze the weights
    model_old.cuda()
    fisher=fisher_matrix_diag(data0,target0,model,criterion,sbatch=100,epoch=0)
    for epoch in range(1, args.epochs + 1):
        args.seed=epoch
        start_time = time.time()
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        if tst_acc>0 and sign:
            args.lr=0.03
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            patience=8
            sign=False
            model_old.load_state_dict(copy.deepcopy(model.state_dict()))
            model_old=model_old.cuda()
        if tst_acc>0.52:
            args.lr=0.03
            args.epsilon+=2/255
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            patience=8
            sign=False
            model_old.load_state_dict(copy.deepcopy(model.state_dict()))
            model_old=model_old.cuda()
        if sign:
            train_adv0(args, model,device, train_loader, optimizer, epoch)
            tstloss, tst_acc,tst_acc_origin = eval_test(model, device, test_loader)
            # evaluation on natural examples    
            tstt.append(tst_acc)
            print('Epoch '+str(epoch)+': '+str(int(time.time()-start_time))+'s', end=', ')
            #print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
            print('test_loss: {:.4f}, test_acc: {:.2f}%,test_acc_origin: {:.2f}%'.format(tstloss, 100. * tst_acc,100.*tst_acc_origin))
            #print('test_adv_loss: {:.4f}, test_adv_acc: {:.2f}%'.format(tst_adv_loss, 100. * tst_adv_acc))
            f=open("result_PREACTresnet18_rand.txt","a")
            f.write('test_loss: '+f'{tstloss}'+','+' '+f'{tst_acc}'+', '+f'{tst_acc_origin}'+'\r\n')
            f.close()
            if tst_acc<max(tstt) and patience!=0:

                model.load_state_dict(copy.deepcopy(model_old.state_dict()))
                print(patience)
                patience-=1
            if patience==0:
                base=epoch
                args.lr=args.lr/3
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                patience=8
            model.train()
            if tst_acc==max(tstt):
                for i in range(0):
                    tstloss, tst_acc0,tst_acc_origin = eval_test(model, device, test_loader)
                    tst_acc+=tst_acc0
                #tst_acc/=3
                if tst_acc>=max(tstt):
                    tstt[-1]=tst_acc
                    patience=8
                    torch.save(model.state_dict(),
                            os.path.join(model_dir, 'wideresnet18_10_no_rand.pt'))
        else:
        # adversarial training
            model_old.load_state_dict(copy.deepcopy(model.state_dict()))
                    #model_old=model_old.cuda()
            freeze_model(model_old) # Freeze the weights
            
            train(args, model,device, train_loader, optimizer, epoch)
            fisher=fisher_matrix_diag(data0,target0,model,criterion,sbatch=100,epoch=epoch-base)
            tstloss, tst_acc,tst_acc_origin = eval_test(model, device, test_loader)
            # evaluation on natural examples    
            tstt.append(tst_acc)
            print('Epoch '+str(epoch)+': '+str(int(time.time()-start_time))+'s', end=', ')
            #print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
            print('test_loss: {:.4f}, test_acc: {:.2f}%,test_acc_origin: {:.2f}%'.format(tstloss, 100. * tst_acc,100.*tst_acc_origin))
            #print('test_adv_loss: {:.4f}, test_adv_acc: {:.2f}%'.format(tst_adv_loss, 100. * tst_adv_acc))
            f=open("result_PREACTresnet18_rand.txt","a")
            f.write('test_loss: '+f'{tstloss}'+','+' '+f'{tst_acc}'+', '+f'{tst_acc_origin}'+'\r\n')
            f.close()
            
            if tst_acc<max(tstt) and patience!=0:
                #model.load_state_dict(copy.deepcopy(model_old.state_dict()))
                print(patience)
                patience-=1
            if patience==0:
                base=epoch
                args.lr=args.lr/3
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                patience=8
           
            model.train()
            #fisher={}
            #fisher=fisher_matrix_diag(data0,target0,model,criterion,sbatch=100)
            # save checkpoint
            if tst_acc==max(tstt):
                if tst_acc>=max(tstt):         
                    patience=8
                    torch.save(model.state_dict(),
                            os.path.join(model_dir, 'epoch'+f"{epoch}"+".pt"))
if __name__ == '__main__':
    config = CONFIGS['ViT-B_16']
    #os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4'
    main()