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
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from distillation import self_distillation
from torch.autograd import Variable
from model_search import Network
from architect import Architect
from metric import performance
import cv2
from imutils import paths
from tensorboardX import SummaryWriter 

parser = argparse.ArgumentParser("cifar-10")
parser.add_argument('--data', type=str, default='/root/SC-new/data/cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.05, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.01, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=25, help='num of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')
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

parser.add_argument('--num_classes', type=int, default=10, help='number of classes')

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
threshold = 0.8


# cpdd
# class ChaLocDataLoader(torch.utils.data.Dataset):
#     def __init__(self, img_dir,imgSize, is_transform=None):
#         self.img_dir = img_dir
#         self.img_paths = []
#         for i in range(len(img_dir)):
#             self.img_paths += [el for el in paths.list_images(img_dir[i])]
#         # self.img_paths = os.listdir(img_dir)
#         # print self.img_paths
#         self.img_size = imgSize
#         self.is_transform = is_transform

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, index):
#         img_name = self.img_paths[index]
#         img = cv2.imread(img_name)
#         resizedImage = cv2.resize(img, self.img_size)
#         resizedImage = np.reshape(resizedImage, (resizedImage.shape[2], resizedImage.shape[0], resizedImage.shape[1]))

#         iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
#         [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]

#         # tps = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
#         # for dot in tps:
#         #     cv2.circle(img, (int(dot[0]), int(dot[1])), 2, (0, 0, 255), 2)
#         # cv2.imwrite("/home/xubb/1_new.jpg", img)

#         ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
#         assert img.shape[0] == 1160
#         new_labels = [(leftUp[0] + rightDown[0])/(2*ori_w), (leftUp[1] + rightDown[1])/(2*ori_h), (rightDown[0]-leftUp[0])/ori_w, (rightDown[1]-leftUp[1])/ori_h]

#         resizedImage = resizedImage.astype('float32')
#         # Y = Y.astype('int8')
#         resizedImage /= 255.0
#         # lbl = img_name.split('.')[0].rsplit('-',1)[-1].split('_')[:-1]
#         # lbl = img_name.split('/')[-1].split('.')[0].rsplit('-',1)[-1]
#         # lbl = map(int, lbl)
#         # lbl2 = [[el] for el in lbl]

#         # resizedImage = torch.from_numpy(resizedImage).float()
#         return resizedImage, new_labels

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
  model.load_state_dict(torch.load('model_cifar10.pt'))
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  # device_2 = torch.device("cuda:1")
  # criterion_2 = nn.CrossEntropyLoss().to(device_2)
  # former_model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion_2, device_2).to(device_2)
  

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  imgSize = (80, 80)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)


#cpdd
  # traindir = os.path.join(args.data, 'train').split(',')
  # validdir = os.path.join(args.data, 'val').split(',')

  # print(traindir)

  # train_dataset = ChaLocDataLoader(traindir, imgSize)
  # valid_dataset = ChaLocDataLoader(validdir, imgSize)

#imagenet
  # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  # train_data = dset.ImageFolder(
  #   traindir,
  #   transforms.Compose([
  #     transforms.RandomResizedCrop(224),
  #     transforms.RandomHorizontalFlip(),
  #     transforms.ColorJitter(
  #       brightness=0.4,
  #       contrast=0.4,
  #       saturation=0.4,
  #       hue=0.2),
  #     transforms.ToTensor(),
  #     normalize,
  #   ]))
  # valid_data = dset.ImageFolder(
  #   validdir,
  #   transforms.Compose([
  #     transforms.Resize(256),
  #     transforms.CenterCrop(224),
  #     transforms.ToTensor(),
  #     normalize,
  #   ]))

#cpdd
  # print("train length: " + str(len(train_dataset)))
  # print("valid length: " + str(len(valid_dataset)))

  # train_queue = torch.utils.data.DataLoader(
  #   train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

  # valid_queue = torch.utils.data.DataLoader(
  #   valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  writer = SummaryWriter(log_dir="/root/tf-logs/train_base")

  dis_ops = 0
  loss_1 = 0
  loss_2 = 0
  t = 0
  arch_param_former = model.arch_parameters()
  arch_final = model.arch_parameters()
  arch_param_nor, arch_param_red = None, None

  # for epoch in range(50):
    
  #   pro_n, pro_r = torch.ones_like(model.alphas_normal), torch.ones_like(model.alphas_reduce)

  #   scheduler.step()
  #   lr = scheduler.get_lr()[0]
  #   logging.info('search epoch %d lr %e', epoch, lr)

  #   genotype = model.genotype()

  #   print(F.softmax(model.alphas_normal, dim=-1))
  #   print(F.softmax(model.alphas_reduce, dim=-1))

  #   # differ_s_t, differ_GT, accuracy = search(former_model, train_queue, valid_queue, model, architect, optimizer, lr, args)

  #   differ_s_t, differ_GT = search(former_model, train_queue, valid_queue, model, architect, optimizer, lr, args)
    
  #   print(differ_s_t)
  #   print(differ_GT)
  #   # print(accuracy)

  #   writer.add_scalar('differ_s_t', differ_s_t, epoch)
  #   writer.add_scalar('differ_GT', differ_GT, epoch)
  #   # writer.add_scalar('accuracy', accuracy, epoch)

  #   if epoch == 0:
  #     arch_param_nor, arch_param_red = model.alphas_normal, model.alphas_reduce
  #     t = differ_s_t

  #   if differ_s_t > t:
  #     arch_param_nor, arch_param_red = model.alphas_normal, model.alphas_reduce
  #     t = differ_s_t

  #   if differ_GT > threshold:
  #     break

  #   if epoch==0:
  #     loss_1 = differ_s_t
  #     loss_2 = differ_GT
        
  #   # if epoch%2==0 and epoch!=0:
  #   else:

  #     similar = differ_s_t - loss_1
  #     perfor = differ_GT - loss_2
  #     loss_1 = differ_s_t
  #     loss_2 = differ_GT

  #     del differ_s_t, differ_GT

  #     if perfor > 0:

  #       if similar > 0:
  #         pro_n, pro_r = discard_space_case1(model.arch_parameters())

  #       else:
  #         if dis_ops < args.dis_ops_max:
  #           pro_n, pro_r = discard_space_case2(model.arch_parameters(), dis_ops)
  #           dis_ops += 1

  #         else:
  #           pass


  #     elif perfor < 0:

  #       if similar > 0:
  #         if dis_ops < args.dis_ops_max:
  #           pro_n, pro_r = discard_space_case3(model.arch_parameters(), dis_ops)
  #           dis_ops += 1

  #         else:
  #           pass

  #       else:
  #         pro_n, pro_r = discard_space_case4(model.arch_parameters())

  #     else:
  #       pass
    
  #   arch_param_former = model.arch_parameters()

  #   former_model.set_arch_params([arch_param_former[0].to(torch.device("cuda:1")),arch_param_former[1].to(torch.device("cuda:1"))])

  #   a_nor = arch_param_nor.detach()
  #   a_red = arch_param_red.detach()
  #   pro_n = pro_n.to(torch.device("cuda:0"))
  #   pro_r = pro_r.to(torch.device("cuda:0"))
  #   a_nor = a_nor.mul_(pro_n)
  #   a_red = a_red.mul_(pro_r)

  #   model.set_arch_params([a_nor.requires_grad_(True),
  #                          a_red.requires_grad_(True)])
    
  #   del pro_n, pro_r
  
  # arch_nor = torch.zeros_like(model.alphas_normal)
  # arch_red = torch.zeros_like(model.alphas_reduce)

  # for i in range(arch_nor.shape[0]):
  #   sec_max = arch_nor[i].sort()[0][-1]
  #   for j in range(arch_nor.shape[1]):
  #     if model.arch_parameters()[0][i][j]>=sec_max:
  #       arch_nor[i][j]=1
    
  # for i in range(arch_red.shape[0]):
  #   sec_max = arch_red[i].sort()[0][-1]
  #   for j in range(arch_red.shape[1]):
  #     if model.arch_parameters()[1][i][j]>=sec_max:
  #       arch_red[i][j]=1

  # a_n_T, a_r_T = arch_final[0].detach(), arch_final[1].detach()
  # a_n_T = a_n_T.mul_(arch_nor)
  # a_r_T = a_r_T.mul_(arch_red)

  # for i in range(a_n_T.shape[0]):
  #   for j in range(a_n_T.shape[1]):
  #     if a_n_T[i][j] == 0:
  #       a_n_T[i][j]=-1e8
    
  # for i in range(a_r_T.shape[0]):
  #   for j in range(a_r_T.shape[1]):
  #     if a_r_T[i][j] == 0:
  #       a_r_T[i][j]=-1e8

# cpdd
  # a_n_T = torch.Tensor([[0., 0., 0., 0., 0., 1., 0., 0.],
  #       [0., 1., 0., 0., 0., 0., 0., 0.],
  #       [0., 0., 0., 0., 0., 1., 0., 0.],
  #       [0., 0., 0., 0., 0., 0., 1., 0.],
  #       [0., 0., 0., 0., 1., 0., 0., 0.],
  #       [0., 0., 0., 0., 0., 1., 0., 0.],
  #       [0., 0., 0., 1., 0., 0., 0., 0.],
  #       [0., 0., 0., 0., 0., 0., 1., 0.],
  #       [0., 0., 0., 0., 1., 0., 0., 0.],
  #       [0., 0., 0., 0., 0., 0., 0., 1.],
  #       [0., 0., 0., 0., 1., 0., 0., 0.],
  #       [0., 0., 1., 0., 0., 0., 0., 0.],
  #       [0., 0., 1., 0., 0., 0., 0., 0.],
  #       [0., 0., 0., 0., 0., 0., 0., 1.]]).to('cuda:0')
  # a_r_T = torch.Tensor([[0., 0., 1., 0., 0., 0., 0., 0.],
  #       [0., 0., 0., 1., 0., 0., 0., 0.],
  #       [0., 0., 0., 1., 0., 0., 0., 0.],
  #       [0., 0., 0., 0., 0., 0., 1., 0.],
  #       [0., 0., 0., 0., 0., 0., 0., 1.],
  #       [0., 0., 0., 0., 0., 0., 0., 1.],
  #       [0., 0., 0., 0., 0., 0., 0., 1.],
  #       [0., 0., 0., 0., 0., 0., 0., 1.],
  #       [0., 0., 1., 0., 0., 0., 0., 0.],
  #       [0., 0., 0., 0., 0., 0., 0., 1.],
  #       [0., 0., 0., 1., 0., 0., 0., 0.],
  #       [0., 0., 0., 0., 0., 1., 0., 0.],
  #       [0., 0., 0., 0., 0., 1., 0., 0.],
  #       [0., 0., 0., 0., 0., 1., 0., 0.]]).to('cuda:0')



  a_n_T = torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000]]).cuda()
  a_r_T = torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000]]).cuda()

  arch_final = [a_n_T, a_r_T]

  model.set_arch_params(arch_final)

  del arch_param_nor, arch_param_red

  # print(F.softmax(model.alphas_normal, dim=-1))
  # print(F.softmax(model.alphas_reduce, dim=-1))
  print("========normal=======")
  print(model.alphas_normal)
  print("========reduce=======")
  print(model.alphas_reduce)
  print("=========end=========")

  a1, a2 = 1, 1

  #training student model
  for epoch in range(args.epochs):

    # if epoch >= 40:
    #   a1 *= 0.95
    #   a2 *= 0.85

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

    torch.save(model.state_dict(), 'model_cifar10.pt')

def search(former_model, train_queue, valid_queue, model, architect, optimizer, lr, args):

  model.train().half()
  former_model.train().half()
  #loss_1, loss_2 = torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)

  arch_nor = torch.zeros_like(model.alphas_normal)
  arch_red = torch.zeros_like(model.alphas_reduce)

  for step, (input, target) in enumerate(train_queue):

    # if step ==1:
    #  break

    # cpdd target = np.array([i.numpy() for i in target]).T

    input = Variable(input, requires_grad=False).cuda().half()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True).half()
    
    # get a random minibatch from the search queue with replacement
    (input_search, target_search) = next(iter(valid_queue))

    # cpddtarget_search = np.array([i.numpy() for i in target_search]).T

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

  # differ_s_t, differ_GT, accuracy = performance(former_model, model, valid_queue, args)

  differ_s_t, differ_GT = performance(former_model, model, valid_queue, args)

  model.set_arch_params(arch_params)
    
  return differ_s_t, differ_GT#, accuracy

def train(train_queue, valid_queue, model, criterion, optimizer, a1, a2):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  model = model.to(torch.float32)
  model.train()
  
  loss_1, loss_2 = torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)

  delta1, delta2 =[],[]

  prec1 = 0

  for step, (input, target) in enumerate(train_queue):


    optimizer.zero_grad()
    
    n = input.size(0)

    # cpdd target = np.array([i.numpy() for i in target]).T

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)
    
    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

    logits = model(input)

    # print('*********output***********')
    # print(logits)
    # print('*********target***********')
    # print(target)
    # print('***********end************')

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
    loss = loss_1 + loss_2
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    if torch.equal(logits,target):
      prec1 += 1 

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      # cpdd logging.info('train %03d %e %f', step, objs.avg, prec1 / n)

  del input
  del target
  del logits
  del loss_l
  del loss_fea
  del loss_s_t

    # 释放未使用的显存
  torch.cuda.empty_cache()


  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  prec1 = 0

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      # cpdd target = np.array([i.numpy() for i in target]).T

      input = Variable(input, requires_grad=False).cuda()
      target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

# cpdd
    # if torch.equal(logits,target):
    #   prec1 += 1 
    # top1.update(prec1.item(), n)
    # top5.update(prec5.item(), n)

    # if step % args.report_freq == 0:
    #   logging.info('train %03d %e %f', step, objs.avg, prec1 / n)
  del input
  del target
  del logits

    # 释放未使用的显存
  torch.cuda.empty_cache()

  return top1.avg, objs.avg

def discard_space_case1(params):
  pro_n=torch.ones(params[0].shape, requires_grad=False)
  pro_r=torch.ones(params[1].shape, requires_grad=False)
  weights_n=F.softmax(params[0], dim=-1)
  weights_r=F.softmax(params[1], dim=-1)

  for i in range(weights_n.shape[0]):
    for j in range(weights_n.shape[1]):
      if params[0][i][j] < 0:
        pro_n[i][j] += 2.1
      elif params[0][i][j] >= 0:
        if weights_n[i][j]<weights_n[i].mean():
          pro_n[i][j] -= 2.1

  for i in range(weights_r.shape[0]):
    for j in range(weights_r.shape[1]):
      if params[1][i][j] < 0:
        pro_r[i][j] += 2.1
      elif params[1][i][j] >= 0:
        if weights_r[i][j]<weights_r[i].mean():
          pro_r[i][j] -= 2.1
  
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
        pro_n[i][j] -= 2.2
      elif params[0][i][j] >= 0:
        if weights_n[i][j]>weights_n[i].mean():
          pro_n[i][j] += 2.5

  for i in range(weights_r.shape[0]):
    for j in range(weights_r.shape[1]):
      if params[1][i][j] < 0:
        pro_r[i][j] += 2.2
      elif params[1][i][j] >= 0:
        if weights_r[i][j]>weights_r[i].mean():
          pro_r[i][j] -= 2.5
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

