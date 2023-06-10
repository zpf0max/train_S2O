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
class timm_model(torch.nn.Module):
    def __init__(self,device,model):
        torch.nn.Module.__init__(self)
        self.device=device
        self.model = model
        self.input_size = self.model.default_cfg['input_size'][1]
        self.interpolation = self.model.default_cfg['interpolation']
        self.crop_pct = self.model.default_cfg['crop_pct']
        self.mean=self.model.default_cfg['mean']
        self.std=self.model.default_cfg['std']
        self.model = self.model.to(self.device)

    def forward(self, x):
        self.mean = torch.tensor(self.mean).view(3,1,1).to(self.device)
        self.std = torch.tensor(self.std).view(3,1,1).to(self.device)   
        x = (x - self.mean) / self.std
        labels = self.model(x.to(self.device))
        return labels

class efficientnet_model(torch.nn.Module):
    def __init__(self,device,model):
        torch.nn.Module.__init__(self)
        self.device=device
        self.model = model
        self.input_size = 224
        self.interpolation = 'bicubic'
        self.mean=(0.485,0.456,0.406)
        self.std=(0.229,0.224,0.225)
        self.model = self.model.to(self.device)

    def forward(self, x):
        self.mean = torch.tensor(self.mean).view(3,1,1).to(self.device)
        self.std = torch.tensor(self.std).view(3,1,1).to(self.device)   
        x = (x - self.mean) / self.std
        labels = self.model(x.to(self.device))
        return labels

def cifar10(batchsize, cifar10_path,input_size,interpolation,transforms): 
    transforms = [transforms.Resize(size=input_size, interpolation=interpolation), transforms.CenterCrop(input_size), transforms.ToTensor()]
    cifar = CIFAR10(root=cifar10_path, train=False, download=True, transform=transforms)
    test_loader = DataLoader(cifar, batch_size=batchsize, shuffle=False, num_workers=1, pin_memory= False, drop_last= False)
    test_loader.name = "cifar10"
    test_loader.batch = batchsize
    return test_loader
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

class ImageNetDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, meta_file, transform=None):

        self.data_dir = data_dir
        self.meta_file = meta_file
        self.transform = transform
        self._indices = []
        for line in open(os.path.join(os.path.dirname(__file__), meta_file), encoding="utf-8"):
            img_path, label, target_label = line.strip().split(' ')
            self._indices.append((os.path.join(self.data_dir, img_path), label, target_label))

    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, index):
        img_path, label, target_label = self._indices[index]
        img = Image.open(img_path).convert('RGB')
        label = int(label)
        target_label=int(target_label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, target_label





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

def get_pred_labels(args, input_tensor, model_name, device, gpu_id):
    if model_name.find('efficientnet')>=0:
        test_net = efficientnet_pytorch.EfficientNet.from_pretrained(model_name)
        test_net = efficientnet_model(device, test_net)
        test_input_size = test_net.input_size
        test_interpolation = test_net.interpolation
    else:
        #set_trace()
        test_net = timm.create_model(model_name, pretrained=True)
        test_net = timm_model(device, test_net)
        test_input_size = test_net.input_size
        test_interpolation = test_net.interpolation
                   
    test_net = test_net
    test_net.eval()
    resized_adv=F.interpolate(input=input_tensor, size=test_input_size, mode=test_interpolation)
    test_out=test_net(resized_adv)
    test_out = torch.argmax(test_out, dim=1)
    return test_out
parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
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
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2/255,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar10-wideresnet/0.3',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=100, type=int, metavar='N',
                    help='save frequency')

parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument("--gpu", type=str, default="cuda:1", help="Comma separated list of GPU ids")
    #parser.add_argument('--dataset_name', default='imagenet', help= 'Dataset for this model', choices= ['cifar10', 'imagenet'])
parser.add_argument('--crop_pct', type=float, default=0.875, help='Input image center crop percent') 
parser.add_argument('--input_size', type=int, default=224, help='Input image size') 
parser.add_argument('--interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic'], help='') 
parser.add_argument('--norm', default=np.inf, help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support cw and deepfool', choices=[np.inf, 2])
parser.add_argument('--data_dir', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/ILSVRC2012_img_val'), help= 'Dataset directory')
parser.add_argument('--label_file', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/val_zpf.txt'), help= 'Dataset directory')
    #parser.add_argument('--data_dir', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/val'), help= 'Dataset directory')
    #parser.add_argument('--label_file', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/imagenet_val.txt'), help= 'Dataset directory')
    #parser.add_argument('--net_name', default='tv_resnet50', help='net_name for sgm', choices= ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--batchsize', default=60, help= 'batchsize for this model')
parser.add_argument('--attack_name', default='pgd', help= 'Dataset for this model', choices= ['fgsm', 'bim', 'pgd','mim','si_ni_fgsm','vmi_fgsm','sgm', 'dim', 'tim', 'deepfool', 'cw','tta'])
    
    # This is what adversrail examples crafted on
    #parser.add_argument('--target_name', default='swin_base_patch4_window7_224', help= 'target model', choices= ['resnet50', 'vgg16', 'inception_v3', 'swin_base_patch4_window7_224'])
parser.add_argument('--target_name', default='vit_small_patch16_224')
    #parser.add_argument('--target_name', default='resnet50')
parser.add_argument('--eps', type= float, default=8/255, help='linf: 8/255.0 and l2: 3.0')
parser.add_argument('--stepsize', type= float, default=2/255, help='linf: 8/2550.0 and l2: (2.5*eps)/steps that is 0.075')
parser.add_argument('--steps', type= int, default=20, help='linf: 100 and l2: 100, steps is set to 100 if attack is apgd')
parser.add_argument('--decay_factor', type= float, default=1.0, help='momentum is used')
parser.add_argument('--resize_rate', type= float, default=0.85, help='dim is used')    #0.9
parser.add_argument('--diversity_prob', type= float, default=0.7, help='dim is used')    #0.5
parser.add_argument('--kernel_name', default='gaussian', help= 'kernel_name for tim', choices= ['gaussian', 'linear', 'uniform'])
parser.add_argument('--len_kernel', type= int, default=15, help='len_kernel for tim')
parser.add_argument('--nsig', type= int, default=3, help='nsig for tim')
parser.add_argument('--scale_factor', type= int, default=1, help='scale_factor for si_ni_fgsm, min 1, max 5')
#parser.add_argument('--beta', type= float, default=1.5, help='beta for vmi_fgsm')
parser.add_argument('--sample_number', type= int, default=10, help='sample_number for vmi_fgsm')
    
parser.add_argument('--overshoot', type= float, default=0.02)
parser.add_argument('--max_steps', type= int, default=50)

parser.add_argument('--loss', default='ce', help= 'loss for fgsm, bim, pgd, mim, dim and tim', choices= ['ce', 'cw'])
parser.add_argument('--kappa', type= float, default=0.0)
#parser.add_argument('--lr', type= float, default=0.2)
parser.add_argument('--init_const', type= float, default=0.01)
parser.add_argument('--binary_search_steps', type= int, default=4)
parser.add_argument('--max_iter', type= int, default=200)
parser.add_argument('--target', default=False, help= 'target for attack', choices= [True, False])

parser.add_argument('--ckpt_netG', type=str, default='attack_benchmark/checkpoints/netG_-1_img_incv3_imagenet_0_rl.pth', help='checkpoint path to netG of CDA')
parser.add_argument('--gk', default=False, help= 'apply Gaussian smoothing to GAN output', choices= [True, False])
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

torch.distributed.init_process_group(
    backend='nccl',
    init_method='env://',
    #timeout=datetime.timedelta(0, 1800),
    world_size=1,
    rank=0,
    store=None,
    group_name=''
)

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
            #loss2 = nn.CrossEntropyLoss()(model(X_pgd)[0], y)
            loss2 = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss2.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    print(1)
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
                           beta=args.beta)
        #loss1 = loss_adv.detach_().requires_grad_(True)
        loss_adv.backward()
        optimizer.step()

def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #data, target = data.cuda(), target.to(device)
            data, target = data.cuda(), target.cuda()
            
            with torch.no_grad():
                X, y = Variable(data, requires_grad=True), Variable(target)
                data_adv = _pgd_whitebox(copy.deepcopy(model), X, y)
            
            output = model(data_adv)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.1
    if epoch >= 150:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # init model, ResNet18() can be also used here for training
    #model=wideResNet(num_classes=10)
    model = WideResNet(num_classes=10)#.to(device)
    #model = ResNet(50,10)
    #pretrain_weights_path = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/checkpoint/cifar10/resnet-50finetune.t7"#resnet50-finetune
    #pretrain_weights_path = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/checkpoint/cifar10/resnet-50.t7"
    #pretrain_weights_path = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/model-cifar10-wideresnet/result_adv_train/resnet50_uncrop.pt"  #resnet_adv_train
    pretrain_weights_path = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/model-cifar10-wideresnet/0.3/epoch103.pt" #wideresnet_adv_train
    #pretrain_weights_path = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/model-cifar10-wideresnet/0.4/epoch128.pt"
    #pretrain_weights_path ='/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/model-cifar10-wideresnet/wide_resnet/epoch19.pt'
    #pretrain_weights_path = "/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/checkpoint/cifar10/wide-resnet-34x10.t7"
    weights = torch.load(pretrain_weights_path)
 
    #print(weights)
    #model=weights
    
    weights_dict = {}
    for k, v in weights.items():
        #for k, v in weights['net']:
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] =v 
    weights_dict0={}
    
    model.load_state_dict(weights_dict)
    
    #model1 = ResNet(50,10)
    #model1 = WideResNet(num_classes=10)#.to(device)
    #weight1=torch.load("/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/checkpoint/cifar10/resnet-50.t7")
    #weight1=torch.load("/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/checkpoint/cifar10/wide-resnet-34x10.t7")
    #model1=weight1['net']
    #model=weights['net']
    model=model.cuda()
 
    #model = torch.nn.parallel.DistributedDataParallel(model,broadcast_buffers=False, find_unused_parameters=True)
    model = torch.nn.DataParallel(model)
     
    #model = PreActResNet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    num_classes = 10
    config = CONFIGS['ViT-B_16']
    #test_net = timm.create_model(args.target_name, pretrained=True,checkpoint_path="/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/output/cifar10-100_500_checkpoint.bin")
    model0 = VisionTransformer(config, 224, zero_head=True, num_classes=num_classes)
    model0.load_state_dict(torch.load("/home/zhaopf/S2O-main/CIFAR10_TRADES_S2O/output/cifar10-100_500_checkpoint.bin"))
    model0.cuda()
    
    
    tstt = []
    torch_resize =torchvision.transforms.Resize([224,224]) 
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        num=0
        succ_num=0
        for data, target in test_loader:
            #print(target.shape)
            batch_size=data.shape[0]
            num+=batch_size
            #data, target = data.cuda(), target.to(device)
            data, target = data.cuda(), target.cuda()
            #data=torch_resize(data)
            with torch.no_grad():
                X, y = Variable(data, requires_grad=True), Variable(target)
                data_adv = _pgd_whitebox(copy.deepcopy(model), X, y)
            #print(data_adv.shape)
            #logits2=model1(data_adv)
            data_adv=torch_resize(data_adv)
            logits = model0(data_adv)
            
            #data=torch_resize(data)
            #logits = model0(data)[0]
            #data=torch_resize(data)
            #logits = model(data)
            preds = torch.argmax(logits, dim=-1)
            #preds2=torch.argmax(logits2, dim=-1)
            '''
            for i in range(batch_size):
                if preds2[i]!=target[i]:
                    num+=1
                    if preds[i]!=target[i]:
                        succ_num+=1
            '''
            succ_num+=(preds!=target).sum()
            #succ_num1+=(preds!=target).sum()
        print("success_rate",succ_num/num)
        # save checkpoint
        '''
        
         if (epoch>99 and epoch%5==0) or (epoch>99 and epoch<110) or (epoch>149 and epoch<160) or (epoch>99 and tstacc==max(tstt)):
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'epoch{}.pt'.format(epoch)))
        '''
       
    
if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4'
    main()