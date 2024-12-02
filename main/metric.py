import torch
import torch.nn as nn 
import torch.nn.functional as F
import utils
from torch.autograd import Variable 
import copy
import numpy as np

__all__ = ['pdist']

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss


def differ_GradCAM(model, final_fea_now, final_fea_former, args):

    difference = []

    weight = model.classifier.weight.data.to(torch.device("cuda:1")).detach()

    for i in range(final_fea_now.shape[0]):
        
        CAM_now = (weight.view(*weight.shape, 1, 1) * final_fea_now[i]).sum(1)
        CAM_now = F.interpolate(CAM_now.unsqueeze(0), size=[32,32], mode='bilinear', align_corners=False)
        CAM_now = F.softmax(CAM_now)

        torch.save(CAM_now, 'attention.pt')
        
        CAM_former = (weight.view(*weight.shape, 1, 1) * final_fea_former[i]).sum(1)
        CAM_former = F.interpolate(CAM_former.unsqueeze(0), size=[32,32], mode='bilinear', align_corners=False)
        CAM_former = F.softmax(CAM_former)
        
        differ = ((CAM_now-CAM_former).sum(dim=[2,3]).squeeze(0)/32**2)
        differ = torch.norm(differ, dim=0, keepdim=True)
        # del CAM_now, CAM_former

        difference.append(differ.to(torch.device("cuda:1")))
        # del differ


    return sum(difference)/len(difference), CAM_now

def performance(former_model, model, valid_queue, args):

    degree = []
    top1_accu = []
    top5_accu = []
    #dist_criterion = RkdDistance()
    
    #dist_list = []
    logits_list_former = []
    logits_list_now = []

    logit_differ_list = []
    distribution_differ = []

    #model_arch_param = model.arch_parameters()

    for _, (input, target) in enumerate(valid_queue):
        
        with torch.no_grad():

            #target_2 = Variable(target).to(torch.device('cuda:1'),non_blocking=True)
            # cpdd  
            # target = np.array([i.numpy() for i in target]).T

            input = Variable(input).cuda().half()
            target = Variable(target).cuda(non_blocking=True).half()

            #input_2 = Variable(input).to(torch.device('cuda:1'))
            #target_2 = Variable(target).to(torch.device('cuda:1'),non_blocking=True)

        torch.save(input, "input.pt")

        logits = model(input)

        #dist_loss = dist_criterion(logits, former(input))

        # prec1_avg, prec5_avg = utils.accuracy(logits, target, (1,5))
        #n = input.size(0)
        # top1_accu.append(prec1_avg.item())
        # top5_accu.append(prec5_avg.item())
        final_fea_now = model.get_final_fea().to(torch.device("cuda:1"))

        #Mr=relation_similarity_metric(former, model, (input, target))
        #Ms=semantic_similarity_metric(former, model, (input, target))

         
        logits_former = former_model(input.to(torch.device("cuda:1")))

        final_fea_former = former_model.get_final_fea()
        

        logits_list_now.append(logits.to(torch.device("cuda:1")).detach())
        logits_list_former.append(logits_former.detach())

        # del logits_former, logits, input

        #dist_list.append(dist_loss.item())

        #model.set_arch_params(model_arch_param)

        M, CAM_now = differ_GradCAM(model, final_fea_now.detach(), final_fea_former.detach(), args)
        # del final_fea_now, final_fea_former
        degree.append(M)

        logits_list_former = torch.stack(logits_list_former).reshape(-1, args.num_classes)
        logits_list_now = torch.stack(logits_list_now).reshape(-1, args.num_classes)


        logits_list_former = torch.norm(logits_list_former, dim=1, keepdim=False)
        logits_list_now = torch.norm(logits_list_now, dim=1, keepdim=False)

        L = torch.unsqueeze(torch.unsqueeze(logits, -1), -1)

        L = torch.mul(L.to(torch.device("cuda:0")), CAM_now.to(torch.device("cuda:0")))
        torch.save(L, 'attention_1.pt')

        min_t, _ = torch.min(target, dim=0)
        max_t, _ = torch.max(target, dim=0)
        for i in range(target.shape[0]):
            target[i] = (target[i] - min_t) / (max_t - min_t)

        logit_differ_list.append((logits_list_now-logits_list_former).mean().squeeze(0))
        distribution_differ.append((logits_list_now-target.to(torch.device("cuda:1"))).mean().squeeze(0))

        logits_list_now = []
        logits_list_former = []
    
    x = sum(degree)[0].item()/len(degree)
    y = sum(logit_differ_list).item()/len(logit_differ_list)
    # z = 0.5*sum(top5_accu)/len(top5_accu)
    u = sum(distribution_differ).item()/len(distribution_differ)
    
    return abs(x) + abs(y), abs(u)#, z
