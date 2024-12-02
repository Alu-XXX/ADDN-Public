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
parser.add_argument('--train_portion', type=float, default=0.9, help='portion of training data')
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

  num_test = len(test_data)
  indices_test = list(range(num_test))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)


  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=96,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test[:]),
      pin_memory=True, num_workers=2)


  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  writer = SummaryWriter(log_dir="runs")

  arch_params = [[[0.1053, 0.1010, 0.1006, 0.1009, 0.2808, 0.1043, 0.1032, 0.1038],
        [0.1147, 0.1123, 0.1106, 0.1114, 0.1142, 0.1140, 0.1140, 0.2087],
        [0.2264, 0.1082, 0.1074, 0.1078, 0.1128, 0.1132, 0.1120, 0.1123],
        [0.1054, 0.1003, 0.0993, 0.1000, 0.2859, 0.1030, 0.1015, 0.1046],
        [0.1091, 0.1034, 0.1032, 0.1037, 0.2524, 0.1092, 0.1090, 0.1100],
        [0.1087, 0.1031, 0.1030, 0.1038, 0.1060, 0.2624, 0.1072, 0.1059],
        [0.1108, 0.1076, 0.1072, 0.1077, 0.1075, 0.1087, 0.2422, 0.1083],
        [0.1078, 0.1038, 0.1037, 0.1038, 0.1094, 0.1052, 0.1077, 0.2587],
        [0.2550, 0.1045, 0.1045, 0.1051, 0.1072, 0.1076, 0.1089, 0.1072],
        [0.1086, 0.1026, 0.1028, 0.1033, 0.2656, 0.1052, 0.1048, 0.1071],
        [0.2786, 0.0998, 0.0997, 0.0999, 0.1057, 0.1057, 0.1062, 0.1045],
        [0.2817, 0.0992, 0.0992, 0.0998, 0.1058, 0.1052, 0.1040, 0.1051],
        [0.2983, 0.0967, 0.0967, 0.0970, 0.1036, 0.1031, 0.1020, 0.1026],
        [0.2897, 0.0981, 0.0980, 0.0981, 0.1019, 0.1048, 0.1044, 0.1049]],
[[0.1161, 0.1163, 0.1139, 0.1149, 0.1905, 0.1169, 0.1154, 0.1159],
        [0.1181, 0.1189, 0.1171, 0.1155, 0.1160, 0.1766, 0.1191, 0.1186],
        [0.1161, 0.1128, 0.1116, 0.1142, 0.2033, 0.1134, 0.1143, 0.1143],
        [0.1118, 0.1146, 0.1141, 0.1148, 0.1110, 0.1136, 0.1130, 0.2071],
        [0.1119, 0.1113, 0.1101, 0.1111, 0.1113, 0.1127, 0.1100, 0.2215],
        [0.1149, 0.1123, 0.1118, 0.2037, 0.1152, 0.1127, 0.1146, 0.1148],
        [0.1064, 0.2486, 0.1106, 0.1067, 0.1071, 0.1055, 0.1063, 0.1088],
        [0.1119, 0.1106, 0.1095, 0.1118, 0.2250, 0.1122, 0.1104, 0.1085],
        [0.1146, 0.1174, 0.1175, 0.1857, 0.1159, 0.1146, 0.1162, 0.1180],
        [0.1083, 0.1066, 0.1061, 0.1083, 0.1065, 0.2438, 0.1096, 0.1109],
        [0.1140, 0.1117, 0.1120, 0.2138, 0.1144, 0.1119, 0.1104, 0.1119],
        [0.1141, 0.1109, 0.1108, 0.1122, 0.1156, 0.1156, 0.2062, 0.1146],
        [0.1120, 0.1104, 0.1111, 0.1122, 0.1115, 0.1155, 0.1156, 0.2117],
        [0.1157, 0.1109, 0.1117, 0.1128, 0.1141, 0.1125, 0.1160, 0.2063]]]
  
  arch_nor = torch.zeros_like(model.alphas_normal)
  arch_red = torch.zeros_like(model.alphas_reduce)
  
  for i in range(arch_nor.shape[0]):
    sec_max = max(arch_params[0][i])
    for j in range(arch_nor.shape[1]):
      if arch_params[0][i][j]>=sec_max:
        arch_nor[i][j]=1
      else:
        arch_nor[i][j]=-1e8
    
  for i in range(arch_red.shape[0]):
    sec_max = max(arch_params[1][i])
    for j in range(arch_red.shape[1]):
      if arch_params[1][i][j]>=sec_max:
        arch_red[i][j]=1
      else:
        arch_red[i][j]=-1e8

  model.set_arch_params([arch_nor, arch_red])


  print(F.softmax(model.alphas_normal, dim=-1))
  print(F.softmax(model.alphas_reduce, dim=-1))

  a1, a2 = 1, 1
  delta1, delta2 = 0, 0

  #training student model
  for epoch in range(90):

    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('train epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)


    train_acc, train_obj, loss_1, loss_2 = train(train_queue, valid_queue, model, criterion_1, optimizer,  a1, a2)
    logging.info('train_acc %f', train_acc)

    writer.add_scalar('train_acc', train_acc, epoch)
    writer.add_scalar('train_loss', train_obj, epoch)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion_1)
    logging.info('valid_acc %f', valid_acc)

    writer.add_scalar('valid_acc', valid_acc, epoch)
    writer.add_scalar('valid_loss', valid_obj, epoch)

    torch.save(model.state_dict(), 'model.pt')

    if epoch==0:

      delta1 = loss_1
      delta2 = loss_2

    else:

      d_1 = loss_1-delta1
      d_2 = loss_2-delta2

      if d_1>0 and d_2<0:
        a1=a1*1.01
        a2=a2*0.99
      elif d_1<0 and d_2>0:
        a1=a1*0.99
        a2=a2*1.01
      elif d_1>0 and d_2>0:
        a1=a1*1.01
        a2=a2*1.01
      elif d_1<0 and d_2<0:
        a1=a1*0.99
        a2=a2*0.99

      delta1 = loss_1
      delta2 = loss_2


  writer.close()

  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  
  model.eval()

  accu = []

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


    accu.append(prec1)

  print(sum(accu)/len(accu))


def train(train_queue, valid_queue, model, criterion, optimizer, a1, a2):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  model = model.to(torch.float32)
  model.train()
  
  loss_1, loss_2 = torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)


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


  return top1.avg, objs.avg, loss_1, loss_2


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

if __name__ == '__main__':
  main() 

