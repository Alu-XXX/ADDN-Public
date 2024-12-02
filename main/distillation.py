import torch
import torch.nn.functional
from torch import nn
import torch.backends.cudnn as cudnn

def kd_loss_function(output, target_output,args):
    output=output/args.temperature
    output_log_softmax=torch.log_softmax(output,dim=1)
    return torch.mean(torch.sum(output_log_softmax*target_output,dim=1))

def feature_loss_function(feature,target_feature):
    loss=(feature-target_feature)**2*((feature>0)|(target_feature>0)).float()
    return torch.abs(loss).sum()/feature.numel()

def self_distillation(model, midden_output, target, midden_net, criterion, ops, args):

    #midden_input initialize
    cudnn.benchmark = True

    midden_output_fea=[]
    
    for i in range(2):
        midden_output_fea.append(ops[0](midden_output[i][1]))
    
    midden_output_fea.append(ops[1](midden_output[2][1]))
    
    for i in range(3, 5):
        midden_output_fea.append(ops[2](midden_output[i][1]))

    for i in range(5,8):
        midden_output_fea.append(midden_output[i][1])

    final_fea=model.get_final_fea()
    middle1_fea, middle2_fea, middle3_fea, middle4_fea, middle5_fea, middle6_fea, middle7_fea=midden_output_fea[0],midden_output_fea[1],midden_output_fea[2],midden_output_fea[3],midden_output_fea[4],midden_output_fea[5],midden_output_fea[6]
    
    middle_output1, middle_output2, middle_output3, middle_output4, middle_output5, middle_output6, middle_output7=midden_net(midden_output_fea[0]),midden_net(midden_output_fea[1]),midden_net(midden_output_fea[2]),midden_net(midden_output_fea[3]),midden_net(midden_output_fea[4]),midden_net(midden_output_fea[5]),midden_net(midden_output_fea[6])


    middle1_loss = criterion(middle_output1, target)
    middle2_loss = criterion(middle_output2, target)
    middle3_loss = criterion(middle_output3, target)
    middle4_loss = criterion(middle_output4, target)
    middle5_loss = criterion(middle_output5, target)
    middle6_loss = criterion(middle_output6, target)
    middle7_loss = criterion(middle_output7, target)
    

    feature_loss_1 = feature_loss_function(middle1_fea, final_fea.detach()) 
    feature_loss_2 = feature_loss_function(middle2_fea, final_fea.detach()) 
    feature_loss_3 = feature_loss_function(middle3_fea, final_fea.detach())   
    feature_loss_4 = feature_loss_function(middle4_fea, final_fea.detach())  
    feature_loss_5 = feature_loss_function(middle5_fea, final_fea.detach()) 
    feature_loss_6 = feature_loss_function(middle6_fea, final_fea.detach())   
    feature_loss_7 = feature_loss_function(middle7_fea, final_fea.detach())

    final_out=model.get_final_out()
    temp4 = final_out / args.temperature
    temp4 = torch.softmax(temp4, dim=1)        

    loss1by4 = kd_loss_function(middle_output1, temp4.detach(), args) * (args.temperature**2)
    loss2by4 = kd_loss_function(middle_output2, temp4.detach(), args) * (args.temperature**2)
    loss3by4 = kd_loss_function(middle_output3, temp4.detach(), args) * (args.temperature**2)
    loss4by4 = kd_loss_function(middle_output4, temp4.detach(), args) * (args.temperature**2)
    loss5by4 = kd_loss_function(middle_output5, temp4.detach(), args) * (args.temperature**2)
    loss6by4 = kd_loss_function(middle_output6, temp4.detach(), args) * (args.temperature**2)
    loss7by4 = kd_loss_function(middle_output7, temp4.detach(), args) * (args.temperature**2)

            
    loss_fea=feature_loss_1+feature_loss_2+feature_loss_3+feature_loss_4+feature_loss_5+feature_loss_6+feature_loss_7
    loss_l=middle1_loss+middle2_loss+middle3_loss+middle4_loss+middle5_loss+middle6_loss+middle7_loss
    loss_s_t=loss1by4+loss2by4+loss3by4+loss4by4+loss5by4+loss6by4+loss7by4

    return loss_l, loss_s_t, loss_fea