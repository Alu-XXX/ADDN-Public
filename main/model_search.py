import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype



class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, is_branch):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier
    self.is_branch=is_branch
    #self.para0=[16,C, C_prev_prev, 1, 1]
    #self.para1=[16,C, C_prev, 1, 1]

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)


  def forward(self, s0, s1, weights):

    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for _ in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)
  
#    input=self.forward(self,s0,s1,weights)
#    block=BottleneckBlock(C,C)
#    output=block(input)
#    return torch.nn.Softmax(torch.nn.Linear(output,C)) 

class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, device_idx, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.midden_output=[]
    self.final_fea=None
    self.final_out=None
    self.device_idx=device_idx


    C_curr = stem_multiplier*C

    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr // 2, C_curr, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )

    # self.stem0 = nn.Sequential(
    #   nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
    #   nn.BatchNorm2d(C // 2),
    #   nn.ReLU(inplace=True),
    #   nn.Conv2d(C // 2, C, kernel_size=3, stride=2, padding=1, bias=False),
    #   nn.BatchNorm2d(C),
    # )

    # self.stem1 = nn.Sequential(
    #   nn.Conv2d(C, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
    #   nn.BatchNorm2d(C // 2),
    #   nn.ReLU(inplace=True),
    #   nn.Conv2d(C // 2, C, kernel_size=3, stride=2, padding=1, bias=False),
    #   nn.BatchNorm2d(C),
    # )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList() 
    reduction_prev = False
    for i in range(layers):
      if i in [layers//4, 2*(layers//4), 3*(layers//4)]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False

      #determinate if distill or not 
      if i%2==0:
        is_branch=True
      else:
        is_branch=False

      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, is_branch)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    # self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.global_pooling = nn.AdaptiveMaxPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    # self.softmax = nn.Softmax()

    self._initialize_alphas()

    # self.net_copy=self.cells
     
  def set_arch_params(self,arch_params):
    self.alphas_normal=arch_params[0]
    self.alphas_reduce=arch_params[1]

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    self.midden_output=[]
    s0 = s1 = self.stem(input)
    # s0 = self.stem0(input)
    # s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        # weights = F.softmax(self.alphas_reduce, dim=-1)
        weights = self.alphas_reduce

      else:
        # weights = F.softmax(self.alphas_normal, dim=-1)
        weights = self.alphas_normal
      s0, s1 = s1, cell(s0, s1, weights)
      if cell.is_branch:
        self.midden_output.append((cell,s1))
    
    # self.net_copy=self.cells
    self.final_fea=s1 
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))

    # logits = self.softmax(logits)

    self.final_out=logits
    return logits


  def get_final_fea(self):
    return self.final_fea
  
  def get_final_out(self):
    return self.final_out

  def _loss(self, input, target):
    logits = self(input)
    #print(logits)

    # print("========logits========")
    # print(logits)
    # print("========target========")
    # print(target)
    # print("==========end=========")


    return self._criterion(logits.float(), target.long()) 
    # return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).to(self.device_idx), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).to(self.device_idx), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

