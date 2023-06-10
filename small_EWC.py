from __future__ import print_function
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
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
from torchvision import models
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
parser.add_argument('--batch_size', type=int, default=30, metavar='N',
                    help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=30, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=16/255,
                    help='perturbation')
parser.add_argument('--num-steps', default=5,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=8/255,
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
parser.add_argument('--model-dir', default='./model-cifar10-wideresnet/result_adv_train',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=100, type=int, metavar='N',
                    help='save frequency')

parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
parser.add_argument('--dataset_name', default='cifar10', help= 'Dataset for this model', choices= ['cifar10', 'imagenet'])
parser.add_argument('--loss', default='ce', help= 'loss for fgsm, bim, pgd, mim, dim and tim', choices= ['ce', 'cw'])
parser.add_argument('--target', default=False, help= 'target for attack', choices= [True, False])
parser.add_argument('--decay_factor', type= float, default=1.0, help='momentum is used')
parser.add_argument('--norm', default=np.inf, help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support cw and deepfool', choices=[np.inf, 2])
parser.add_argument('--net_name', default='resnet50', help='net_name for sgm', choices= ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--batchsize', default=10, help= 'batchsize for this model')
parser.add_argument('--attack_name', default='dim', help= 'Dataset for this model', choices= ['fgsm', 'bim', 'pgd','mim','si_ni_fgsm','vmi_fgsm','sgm', 'dim', 'tim', 'deepfool', 'cw','tta'])
parser.add_argument('--resize_rate', type= float, default=0.85, help='dim is used')    #0.9
parser.add_argument('--diversity_prob', type= float, default=0.7, help='dim is used')    #0.5
    # This is what adversrail examples crafted on
    #parser.add_argument('--target_name', default='swin_base_patch4_window7_224', help= 'target model', choices= ['resnet50', 'vgg16', 'inception_v3', 'swin_base_patch4_window7_224'])
parser.add_argument('--target_name', default='vit_small_patch16_224')
    #parser.add_argument('--target_name', default='resnet50')
parser.add_argument('--eps', type= float, default=16/255, help='linf: 8/255.0 and l2: 3.0')
parser.add_argument('--stepsize', type= float, default=2/255, help='linf: 8/2550.0 and l2: (2.5*eps)/steps that is 0.075')
parser.add_argument('--steps', type= int, default=10, help='linf: 100 and l2: 100, steps is set to 100 if attack is apgd')


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
    transforms.RandomCrop(32),#, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Resize([224,224]) ,
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.5083, 0.5011, 0.5383)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize([224,224]),
    #transforms.Normalize((0.4942, 0.4851, 0.4504), (0.5229, 0.5148, 0.5545)),
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

def criterion(output,targets,fisher2,sbatch):
        # Regularization for all previous tasks
    loss_reg=0
    for (name,param),(_,param_old) in zip(model.named_parameters(),model_old.named_parameters()):
        #print(loss_reg.is_cuda)
        loss_reg+=torch.sum(fisher2[name]*(param_old-param).pow(2))/2
    
    return ce(output,targets)+100000/sbatch*loss_reg
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
        loss=criterion(outputs,target,fisher_old,sbatch)
        loss.backward()
        # Get gradients
        num+=1
        for n,p in model.named_parameters():
            if p.grad is not None:   
                fisher[n]+=sbatch*p.grad.data.pow(2) 
    # Mean          return
    
    for n,p in model.named_parameters():
        fisher[n]=fisher[n]/x.size(0)
        fisher[n]=(fisher_old[n]*(epoch+500)+fisher[n])/(epoch+501)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher
def fisher_matrix_diag2(train_loader,model,criterion,sbatch=20,epoch=0):

    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    fisher_old=copy.deepcopy(fisher)
    '''
    model.train()
    for i,(images,target) in enumerate(train_loader):
        loss=0
        images,target=images.cuda(),target.cuda()
        batchsize=images.shape[0]
        #b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
        #print(b)
        #images=torch.autograd.Variable(x[b],volatile=False)
        #target=torch.autograd.Variable(y[b],volatile=False)
        # Forward and backward
        model.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,target,fisher_old,sbatch=batchsize)
        loss.backward()
        for n,p in model.named_parameters():
            if p.grad is not None:   
                fisher[n]+=batchsize*p.grad.data.pow(2) 
    for n,p in model.named_parameters():
        fisher[n]=fisher[n]/len(train_loader)
        #fisher[n]=(fisher_old[n]*t+fisher[n])/(t+1)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    '''
    return fisher
ATTACKS = {
    'fgsm': FGSM,
    'bim': BIM,
    'pgd': PGD,
    'mim': MIM,
    'cw': CW,
    'deepfool': DeepFool,
    'dim': DI2FGSM,
    'tim': TIFGSM,
    'si_ni_fgsm': SI_NI_FGSM,
    'vmi_fgsm': VMI_fgsm,
    'sgm': SGM,
    'cda': CDA,
    'tta': TTA
}    

def generate_attacker(args, net, device):
    if args.attack_name == 'fgsm':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, p=args.norm, eps=args.eps, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'bim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, p=args.norm, stepsize=args.stepsize, steps=args.steps, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'pgd':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, norm=args.norm, stepsize=args.stepsize, steps=args.steps, data_name=args.dataset_name,target=args.target, loss=args.loss,device=device)
    elif args.attack_name == 'mim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, p=args.norm, stepsize=args.stepsize, steps=args.steps, decay_factor=args.decay_factor, 
        data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'cw':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, device,args.norm, args.target, args.kappa, args.lr, args.init_const, args.max_iter, 
                                args.binary_search_steps, args.dataset_name)
    elif args.attack_name == 'deepfool':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, args.overshoot, args.max_steps, args.norm, args.target, device)
    elif args.attack_name == 'dim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, p=args.norm, eps=args.eps, stepsize=args.stepsize, steps=args.steps, decay=args.decay_factor, 
                            resize_rate=args.resize_rate, diversity_prob=args.diversity_prob, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'tim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, p=args.norm, kernel_name=args.kernel_name, len_kernel=args.len_kernel, nsig=args.nsig, 
                            eps=args.eps, stepsize=args.stepsize, steps=args.steps, decay=args.decay_factor, resize_rate=args.resize_rate, 
                            diversity_prob=args.diversity_prob, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)  
    elif args.attack_name == 'si_ni_fgsm':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, p=args.norm, scale_factor=args.scale_factor, stepsize=args.stepsize, decay_factor=args.decay_factor, steps=args.steps, 
        data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'vmi_fgsm':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, p=args.norm, beta=args.beta, sample_number = args.sample_number, stepsize=args.stepsize, decay_factor=args.decay_factor, steps=args.steps, 
        data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'sgm':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, args.net_name, args.eps, args.norm, args.stepsize, args.steps, gamma=0.0, momentum=args.decay_factor, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name=='cda':
        attack_class=ATTACKS[args.attack_name]
        attack=attack_class(net, args.gk, p=args.norm, eps=args.eps, device=device)
    elif args.attack_name=='tta':
        attack_class=ATTACKS[args.attack_name]
        attack=attack_class(net, epsilon=args.eps, norm=args.norm, stepsize=args.stepsize, steps=args.steps, kernel_size=5, nsig=3, resize_rate=args.resize_rate, diversity_prob=args.diversity_prob, data_name=args.dataset_name,target=args.target, loss=args.loss,device=device)
    return attack
   
   
def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    correct2= 0
    torch_resize =torchvision.transforms.Resize([224,224]) 
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()  
            with torch.no_grad():
                X, y = Variable(data, requires_grad=True), Variable(target)
            data=torch_resize(data)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
 
    return test_loss, test_accuracy
def train(args,model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    
    for param in model_old.parameters():
        param.requires_grad = True
    #fisher_old=copy.deepcopy(fisher)
    torch_resize =torchvision.transforms.Resize([224,224]) 
    tstt=torch.FloatTensor([0]*35)
    #tstt=[0]*20
    tstt=tstt.cuda()
    
    #print(tstt)
    num=0
    train_loader2 = torch.utils.data.DataLoader(trainset , batch_size=10,shuffle=True, **kwargs)
    fisher=fisher_matrix_diag2(train_loader2,model,criterion,sbatch=40,epoch=0)
    #succ_num=0
    delta=torch.FloatTensor([0]*35)
    delta=delta.cuda()
    for batch_idx, (data, target) in enumerate(train_loader):
        model.load_state_dict(copy.deepcopy(model_old.state_dict()))
        
        data, target = data.cuda(), target.cuda()
        #print(data[0][0][0])
        attack=generate_attacker(args, model, device)
        
        data_adv=attack.forward(data, target, None)
        #print(data[0][0][0])
        delta[0]+=(torch.mean(torch.abs(data_adv-data)))
        data_adv=torch_resize(data_adv)
                #outputs=model(data_adv)
                #outputs=torch.argmax(outputs, dim=1)
                #succ_num+=(outputs!=target).sum()
                
        outputs2=model0(data_adv)
        outputs2=torch.argmax(outputs2, dim=1)
        tstt[0]+=(outputs2!=target).sum()
        
        args.lr=0.00005
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        patience=10
        #tstloss, tst_acc = eval_test(model, device, test_loader)
        #print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(tstloss, 100. * tst_acc))
        batch_size=data.shape[0]
        num+=batch_size
        
        for kl in range(1,61):
            '''
            if kl%5==0:
                fisher=fisher_matrix_diag(data0,target0,model,criterion,sbatch=800,epoch=0)
            '''
           
             
            attack=generate_attacker(args, model, device)
            start_time = time.time()
            optimizer.zero_grad()

            
            loss_adv = trades_loss(model=model,
                            model_old=model_old,
                            fisher_old=fisher,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta,
                          )
            
            '''     
            with torch.no_grad():
                X, y = Variable(data, requires_grad=True), Variable(target)
                data_adv = _pgd_whitebox(copy.deepcopy(model_old), X, y)
            outputs=model(data_adv)
            loss_adv=criterion(outputs,target,fisher,sbatch=40)
            '''
            loss_adv.backward()
            optimizer.step()
            if kl%2==0:
                data_adv=attack.forward(data, target, None)
                delta[kl//5]+=(torch.mean(torch.abs(data_adv-data)))
                data_adv=torch_resize(data_adv)
                #outputs=model(data_adv)
                #outputs=torch.argmax(outputs, dim=1)
                #succ_num+=(outputs!=target).sum()
                
                outputs2=model0(data_adv)
                outputs2=torch.argmax(outputs2, dim=1)
                tstt[kl//2]+=(outputs2!=target).sum()
            
            #print("adv_acc",(outputs2==target).sum()/batch_size)
            #tstloss, tst_acc = eval_test(model, device, test_loader)
            # evaluation on natural examples    
            #tstt.append(tst_acc)
            #print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(tstloss, 100. * tst_acc))
            #print('Epoch '+str(kl)+': '+str(int(time.time()-start_time))+'s', end=', ')
            #fisher=fisher_matrix_diag(data0,target0,model,criterion,sbatch=400,epoch=kl)
            '''
            if tst_acc<max(tstt) and patience!=0:
                print(patience)
                patience-=0
            if patience==0:
                args.lr=args.lr/3
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                patience=10
            '''
        print(tstt/num)
        #print(succ_num/num)
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
    global data0
    global target0
    global model0
    #global model0
    ce=torch.nn.CrossEntropyLoss()
    #model0 = WideResNet(num_classes=10)#.to(device)
    #model_old = WideResNet(num_classes=10)#.to(device)
    model=ResNet(50, num_classes)
    model_old=ResNet(50, num_classes)
    #model=PreActResNet18(num_classes=10)
    #model0=PreActResNet18(num_classes=10)
    
    config = CONFIGS['ViT-B_16']
    #test_net = timm.create_model(args.target_name, pretrained=True,checkpoint_path="/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/output/cifar10-100_500_checkpoint.bin")
    model0 = VisionTransformer(config, 224, zero_head=True, num_classes=num_classes)
    model0.load_state_dict(torch.load("/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/output/cifar10-100_500_checkpoint.bin"))
    model0.cuda()
    #model0=VisionTransformer(config, 224, zero_head=True, num_classes=num_classes)
    sign=True
    
    #pretrain_weights_path ="/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/checkpoint/cifar10/resnet-50origin-normalize.t7"
    #pretrain_weights_path = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/model-cifar10-wideresnet/wide_resnet/epoch39.pt"
    #pretrain_weights_path = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/model-cifar10-wideresnet/result_adv_train/resnet50_nomalize.pt"
    pretrain_weights_path = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/checkpoint/cifar10/resnet-50origin-no.t7"
    #pretrain_weights_path = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/model-cifar10-wideresnet/0.4/epoch128.pt"
    weights= torch.load(pretrain_weights_path)

    '''
    weights_dict = {}
    for k, v in weights.items():
        #for k, v in weights['net']:
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] =v     
    model.load_state_dict(weights_dict)
    
    '''
    '''
    model = models.resnet50(pretrained=False)
    model_old = models.resnet50(pretrained=False)
    #net_dict = model.state_dict()
    predict_model = torch.load('/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/checkpoint/cifar10/resnet-50origin-normalize_resize_resize.t7')
    #state_dict = {k: v for k, v in predict_model['net'].items() if k in net_dict.keys()}# 寻找网络中公共层，并保留预训练参数
    #net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
    model=predict_model['net']

        # 修改最后一层全连接层的数量，改为分类种类的数量
    num_ftrs =model_old.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 10)
    #model=weights['net']
    model=model.cuda()
    model_old.fc = nn.Linear(num_ftrs, 10)
    '''
    model=weights['net']
    model.cuda()
    model_old.load_state_dict(copy.deepcopy(model.state_dict()))
    model_old=model_old.cuda()
    #model = torch.nn.parallel.DistributedDataParallel(model,broadcast_buffers=False, find_unused_parameters=True)
    #model = torch.nn.DataParallel(model)
    #model_old = torch.nn.DataParallel(model_old)
    #model = PreActResNet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    data0=torch.tensor([])
    target0=torch.LongTensor([])
    data0, target0= data0.cuda(), target0.cuda()
    train_loader2 = torch.utils.data.DataLoader(trainset , batch_size=800,shuffle=True, **kwargs)
    
    for i,(data,target) in enumerate(train_loader2):
        data, target = data.cuda(), target.cuda()
        data0=torch.cat((data0,data.clone()),dim=0)  
        target0=torch.cat((target0,target.clone()),dim=0)
  
    tstloss, tst_acc = eval_test(model0, device, test_loader)
    print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(tstloss, 100. * tst_acc))
    #tstloss, tst_acc = eval_test(model, device, test_loader)
    #fisher=fisher_matrix_diag(data0,target0,model,criterion,sbatch=200)
    #fisher=fisher_matrix_diag2(train_loader,model,criterion,sbatch=200)
    train(args, model,device, train_loader, optimizer,args.epochs)
    
if __name__ == '__main__':
    config = CONFIGS['ViT-B_16']
    #os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4'
    main()