import copy
import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from distillation import self_distillation
from torch.autograd import Variable
from model_search import Network
from architect import Architect
from metric import performance

from tensorboardX import SummaryWriter 

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/root/autodl-tmp/self-kd+nas/data/cifar100', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=156, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.05, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.01, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.25, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=5e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=3e-3, help='weight decay for arch encoding')
parser.add_argument('--dis_ops_max', type=int, default=7, help='weight decay for arch encoding')
parser.add_argument('--summary_folder', type=str, default='run_out', help='older to save the summary')
parser.add_argument('--resume', default='', type=str,help='path to  latest checkpoint (default: None)')
parser.add_argument('--temperature', type=float, default=5, help='temperature to smooth the logits')
parser.add_argument('--alpha', default=0.1, type=float,help='weight of kd loss')
parser.add_argument('--beta', default=1e-6, type=float,help='weight of feature loss')
parser.add_argument('--start-epoch', default=0, type=int,help='manual iter number (useful on restarts)')
parser.add_argument('--print_freq', default=100, type=int,help='print frequency (default: 10)')
parser.add_argument('--step_ratio', default=0.1, type=float,help='ratio for learning rate deduction')
parser.add_argument('--warm-up', action='store_true',help='for n = 18, the model needs to warm up for 400 ' 'iterations')

parser.add_argument('--num_classes', type=int, default=100, help='number of classes')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = args.num_classes
threshold = 0.5


#def compute_grad(model,loss):
#  grads = torch.autograd.grad(loss, model.parameters(),retain_graph=True,allow_unused=True)
#  return grads[0].data

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion_1 = nn.CrossEntropyLoss().to(args.gpu)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion_1, args.gpu).to(args.gpu)

  model.load_state_dict(torch.load('model.pt'))

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils.cifar100_dataset(args)
  train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=valid_transform)
  test_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  #train_queue = torch.utils.data.DataLoader(
  #    train_data, batch_size=args.batch_size,
  #    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
  #    pin_memory=True, num_workers=2)

  #valid_queue = torch.utils.data.DataLoader(
  #    valid_data, batch_size=args.batch_size,
  #    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
  #    pin_memory=True, num_workers=2)
  
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:]),
      pin_memory=True, num_workers=2)

  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:]),
      pin_memory=True, num_workers=2)


  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  writer = SummaryWriter(log_dir="runs")


  print(F.softmax(model.alphas_normal, dim=-1))
  print(F.softmax(model.alphas_reduce, dim=-1))

  a1, a2 = 1, 1

  #training student model
  for epoch in range(args.epochs+150):

    if epoch >= 40:
      a1 *= 0.95
      a2 *= 0.85

    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('train epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)


    train_acc, train_obj = train(train_queue, valid_queue, model, criterion_1, optimizer,  a1, a2)
    logging.info('train_acc %f', train_acc)

    writer.add_scalar('train_acc', train_acc, epoch)
    writer.add_scalar('train_loss', train_obj, epoch)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion_1)
    logging.info('valid_acc %f', valid_acc)

    writer.add_scalar('valid_acc', valid_acc, epoch)
    writer.add_scalar('valid_loss', valid_obj, epoch)

    if train_acc >= 95 and valid_acc >= 78:
      break

    torch.save(model.state_dict(), 'model.pt')



  writer.close()

  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  
  model.eval()

  for _, (input, target) in enumerate(test_queue):

    with torch.no_grad():
      input = Variable(input).cuda()
      target = Variable(target).cuda(non_blocking=True)

    logits = model(input)
    loss = criterion_1(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)


    logging.info('test_acc %f', prec1)


      

def search(former_model, train_queue, valid_queue, model, architect, optimizer, lr, args):

  model.train().half()
  former_model.train().half()
  #loss_1, loss_2 = torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)

  arch_nor = torch.zeros_like(model.alphas_normal)
  arch_red = torch.zeros_like(model.alphas_reduce)

  for _, (input, target) in enumerate(train_queue):

    #if step ==1:
    #  break

    input = Variable(input, requires_grad=False).cuda().half()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True).half()
    
    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda().half()
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True).half()


    architect.step(input, target, input_search, target_search, lr, optimizer, args.unrolled)

    del input, target, input_search, target_search

  arch_params = [model.alphas_normal.detach().clone(), model.alphas_reduce.detach().clone()]

  for i in range(arch_nor.shape[0]):
    sec_max = arch_params[0][i].sort()[0][-1]
    for j in range(arch_nor.shape[1]):
      if arch_params[0][i][j]>=sec_max:
        arch_nor[i][j]=1
    
  for i in range(arch_red.shape[0]):
    sec_max = arch_params[1][i].sort()[0][-1]
    for j in range(arch_red.shape[1]):
      if arch_params[1][i][j]>=sec_max:
        arch_red[i][j]=1

  mapping(model, arch_nor, arch_red)

  del arch_nor, arch_red

  differ_s_t, differ_GT, accuracy = performance(former_model, model, valid_queue, args)

  model.set_arch_params(arch_params)
    
  return differ_s_t, differ_GT, accuracy

def train(train_queue, valid_queue, model, criterion, optimizer, a1, a2):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  model = model.to(torch.float32)
  model.train()
  
  loss_1, loss_2 = torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)

  delta1, delta2 =[],[]

  for step, (input, target) in enumerate(train_queue):


    optimizer.zero_grad()
    
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)
    
    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

    logits = model(input)

    loss_l, loss_fea, loss_s_t = self_distillation(model, model.midden_output, target, args)
    #loss_1 = loss_fea/4+loss_s_t/4
    #loss_2 = (criterion(logits, target)+loss_l)/5

    loss_1 = loss_fea+loss_s_t
    loss_2 = (criterion(logits, target)+loss_l)
    
    #if step==0:
    #  delta1.append(loss_1)
    #  delta2.append(loss_2)
    #  loss = a1 * loss_1 + a2 * loss_2
    #  loss.backward()

    #else:
    #  delta1[0]=loss_1-delta1[0]
    #  delta2[0]=loss_2-delta2[0]


    #  if delta1[0]>0 and delta2[0]<0:
    #    a1=a1*1.01
    #    a2=a2*0.99
    #  elif delta1[0]<0 and delta2[0]>0:
    #    a1=a1*0.99
    #    a2=a2*1.01
    #  elif delta1[0]>0 and delta2[0]>0:
    #    a1=a1*1.01
    #    a2=a2*1.01
    #  elif delta1[0]<0 and delta2[0]<0:
    #    a1=a1*0.99
    #    a2=a2*0.99

    #  loss = a1 * loss_1 + a2 * loss_2
    #  loss.backward()
    loss = a1 * loss_1 + a2 * loss_2
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)


  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input = Variable(input).cuda()
      target = Variable(target).cuda(non_blocking=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def discard_space_case1(params):
  pro_n=torch.ones(params[0].shape, requires_grad=False)
  pro_r=torch.ones(params[1].shape, requires_grad=False)
  weights_n=F.softmax(params[0], dim=-1)
  weights_r=F.softmax(params[1], dim=-1)

  for i in range(weights_n.shape[0]):
    for j in range(weights_n.shape[1]):
      if params[0][i][j] < 0:
        pro_n[i][j] *= 10
      elif params[0][i][j] >= 0:
        if weights_n[i][j]<weights_n[i].mean():
          pro_n[i][j] /= 10

  for i in range(weights_r.shape[0]):
    for j in range(weights_r.shape[1]):
      if params[1][i][j] < 0:
        pro_r[i][j] *= 10
      elif params[1][i][j] >= 0:
        if weights_r[i][j]<weights_r[i].mean():
          pro_r[i][j] /= 10
  
  return pro_n, pro_r

def discard_space_case2(params, dis_ops):
  pro_n=torch.ones(params[0].shape)
  pro_r=torch.ones(params[1].shape)
  normal=F.softmax(params[0], dim=-1)
  reduce=F.softmax(params[1], dim=-1)

  for i in range(normal.shape[0]):
    discard_num = 0
    sec_max = normal[i].sort()[0][-dis_ops]
    for j in range(normal.shape[1]):
      if discard_num >= dis_ops:
        break
      if normal[i][j] >= sec_max:
        pro_n[i][j] = 0
        discard_num += 1

  for i in range(reduce.shape[0]):
    discard_num = 0
    sec_max = reduce[i].sort()[0][-dis_ops]
    for j in range(reduce.shape[1]):
      if discard_num >= dis_ops:
        break
      if reduce[i][j] >= sec_max:
        pro_r[i][j] = 0
        discard_num += 1

  return pro_n, pro_r


def discard_space_case3(params, dis_ops):
  pro_n=torch.ones(params[0].shape)
  pro_r=torch.ones(params[1].shape)
  normal=F.softmax(params[0], dim=-1)
  reduce=F.softmax(params[1], dim=-1)

  for i in range(normal.shape[0]):
    discard_num = 0
    sec_max = normal[i].sort()[0][-dis_ops]
    for j in range(normal.shape[1]):
      if discard_num >= dis_ops:
        break
      if normal[i][j] >= sec_max:
        pro_n[i][j] = 0
        discard_num += 1

  for i in range(reduce.shape[0]):
    discard_num = 0
    sec_max = reduce[i].sort()[0][-dis_ops]
    for j in range(reduce.shape[1]):
      if discard_num >= dis_ops:
        break
      if reduce[i][j] >= sec_max:
        pro_r[i][j] = 0
        discard_num += 1

  return pro_n, pro_r


def discard_space_case4(params):
  pro_n=torch.ones(params[0].shape, requires_grad=False)
  pro_r=torch.ones(params[1].shape, requires_grad=False)
  weights_n=F.softmax(params[0], dim=-1)
  weights_r=F.softmax(params[1], dim=-1)

  for i in range(weights_n.shape[0]):
    for j in range(weights_n.shape[1]):
      if params[0][i][j] < 0:
        pro_n[i][j] /= 20
      elif params[0][i][j] >= 0:
        if weights_n[i][j]>weights_n[i].mean():
          pro_n[i][j] *= 15

  for i in range(weights_r.shape[0]):
    for j in range(weights_r.shape[1]):
      if params[1][i][j] < 0:
        pro_r[i][j] /= 20
      elif params[1][i][j] >= 0:
        if weights_r[i][j]>weights_r[i].mean():
          pro_r[i][j] *= 15
  return pro_n, pro_r

def mapping(model, mask_n, mask_r):

  arch_params = model.arch_parameters()
  arch_n, arch_r = arch_params[0].detach(), arch_params[1].detach()
  arch_n, arch_r = arch_n.mul_(mask_n), arch_r.mul_(mask_r)
  for i in range(arch_n.shape[0]):
    for j in range(arch_n.shape[1]):
      if arch_n[i][j] == 0:
        arch_n[i][j] = -1e8
  
  for i in range(arch_r.shape[0]):
    for j in range(arch_r.shape[1]):
      if arch_r[i][j] == 0:
        arch_r[i][j] = -1e8

  arch_params = [arch_n, arch_r]
  arch_params[0].requires_grad_(True)
  arch_params[1].requires_grad_(True)
  model.set_arch_params(arch_params)

if __name__ == '__main__':
  main() 

