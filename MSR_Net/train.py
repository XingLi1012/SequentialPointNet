# -*- coding: utf-8 -*-
import torch
import os
import tqdm
import shutil
import collections
import argparse
import random
import time
#import gpu_utils as g
import numpy as np

from model import PointNet_Plus#,Attension_Point,TVLAD
from dataset import NTU_RGBD
from utils import group_points_4DV_T_S

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
def main(args=None):
    parser = argparse.ArgumentParser(description = "Training")

    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')#￥￥￥￥
    parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 3,  help='number of input point features')
    parser.add_argument('--temperal_num', type=int, default = 3,  help='number of input point features')
    parser.add_argument('--pooling', type=str, default='concatenation', help='how to aggregate temporal split features: vlad | concatenation | bilinear')
    parser.add_argument('--dataset', type=str, default='ntu60', help='how to aggregate temporal split features: ntu120 | ntu60')

    parser.add_argument('--weight_decay', type=float, default=0.0008, help='weight decay (SGD only)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')#￥￥￥￥
    parser.add_argument('--gamma', type=float, default=0.5, help='')#￥￥￥￥
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')

    parser.add_argument('--root_path', type=str, default='C:\\Users\\Administrator\\Desktop\\LX\\paper\\dataset\\Prosessed_dataset\\01_MSR3D',  help='preprocess folder')
    # parser.add_argument('--depth_path', type=str, default='C:\\Users\\Administrator\\Desktop\\LX\paper\\dataset\\Prosessed_dataset\\01_MSR3D\\',  help='raw_depth_png')
    ################
    # parser.add_argument('--save_root_dir', type=str, default='C:\\Users\\Administrator\\Desktop\\LX\\paper\\code\\3DV-Action-master\\models\\ntu60\\xsub',  help='output folder')
    parser.add_argument('--save_root_dir', type=str, default='C:\\Users\\Administrator\\Desktop\\LX\\paper\\code\\Models_Parameter\\03_3DV-Action-master(base-run-version)(FPS512-64-K32)(2层局部+时序池化3层简单)(t=6_stride=2_KC=64)(单时序流4维特征)(单流读取)\\models\\msr\\xsub',  help='output folder')
    parser.add_argument('--model', type=str, default = '',  help='model name for training resume')
    parser.add_argument('--optimizer', type=str, default = '',  help='optimizer name for training resume')
    
    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python train.py

    ########
    parser.add_argument('--Seg_size', type=int, default =1,  help='number of frame in seg')
    parser.add_argument('--stride', type=int, default = 1,  help='stride of seg')
    parser.add_argument('--all_framenum', type=int, default = 24,  help='number of action frame')
    parser.add_argument('--framenum', type=int, default = 24,  help='number of action frame')
    parser.add_argument('--EACH_FRAME_SAMPLE_NUM', type=int, default = 512,  help='number of sample points in each frame')
    parser.add_argument('--T_knn_K', type=int, default = 48,  help='K for knn search of temperal stream')
    parser.add_argument('--T_knn_K2', type=int, default = 16,  help='K for knn search of temperal stream')
    parser.add_argument('--T_sample_num_level1', type=int, default = 128,  help='number of first layer groups')
    parser.add_argument('--T_sample_num_level2', type=int, default = 32,  help='number of first layer groups')
    parser.add_argument('--T_ball_radius', type=float, default=0.2, help='square of radius for ball query of temperal stream')
    
    parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')

    parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
    parser.add_argument('--SAMPLE_NUM', type=int, default = 2048,  help='number of sample points')

    parser.add_argument('--Num_Class', type=int, default = 20,  help='number of outputs')
    parser.add_argument('--knn_K', type=int, default = 64,  help='K for knn search')
    parser.add_argument('--sample_num_level1', type=int, default = 512,  help='number of first layer groups')
    parser.add_argument('--sample_num_level2', type=int, default = 128,  help='number of second layer groups')
    parser.add_argument('--ball_radius', type=float, default=0.1, help='square of radius for ball query in level 1')#0.025 -> 0.05 for detph
    parser.add_argument('--ball_radius2', type=float, default=0.2, help='square of radius for ball query in level 2')# 0.08 -> 0.01 for depth


    opt = parser.parse_args()
    print (opt)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=os.path.join(opt.save_root_dir, 'train00.log'), level=logging.INFO)
    # torch.cuda.set_device(opt.main_gpu)

    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    try:
        os.makedirs(opt.save_root_dir)
    except OSError:
        pass

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    ##############################
    data_train = NTU_RGBD(root_path = opt.root_path,opt=opt,
        DATA_CROSS_VIEW = False,
        full_train = True,
        validation = False,
        test = False,
        Transform = True
        )
    train_loader = DataLoader(dataset = data_train, batch_size = opt.batchSize, shuffle = True, drop_last = True,num_workers = 8)
    data_val = NTU_RGBD(root_path = opt.root_path, opt=opt,
        DATA_CROSS_VIEW = False,
        full_train = False,
        validation = False,
        test = True,
        Transform = False
        )
    val_loader = DataLoader(dataset = data_val, batch_size = 24,num_workers = 8)

    netR = PointNet_Plus(opt)

    netR = torch.nn.DataParallel(netR).cuda()
    netR.cuda()
    print(netR)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(netR.parameters(), lr=opt.learning_rate, betas = (0.5, 0.999), eps=1e-06)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=opt.gamma)

    for epoch in range(opt.nepoch):
        scheduler.step(epoch)
        
        # switch to train mode
        torch.cuda.synchronize()
        netR.train()
        acc = 0.0
        loss_sigma = 0.0
        total1 = 0.0
        timer = time.time()
        
        for i, data in enumerate(tqdm(train_loader, 0)):
            if len(data[0])==1:
                continue
            torch.cuda.synchronize()
            # 1 load imputs and target
            ## 3DV points and 3 temporal segment appearance points
            ## points_xyzc: B*2048*8;points_1xyz:B*2048*3  target: B*1
            points4DV_T,label,v_name = data
            points4DV_T,label = points4DV_T.cuda(),label.cuda()
            # print('points4DV_T:',points4DV_T.shape)
            xt, yt = group_points_4DV_T_S(points4DV_T, opt)#B*F*4*Cen*K  B*F*4*Cen*1
            # print('xt:',xt.shape)
            xt = xt.type(torch.FloatTensor)
            yt = yt.type(torch.FloatTensor)

            prediction = netR(xt,yt)

            loss = criterion(prediction,label)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            # update training error
            loss_sigma += loss.item()
            #_, predicted60 = torch.max(prediction.data[:,0:60], 1)
            _, predicted = torch.max(prediction.data, 1)
            # print(predicted.data)
            acc += (predicted==label).cpu().sum().numpy()
            total1 += label.size(0)

        
        acc_avg = acc/total1
        loss_avg = loss_sigma/total1
        print('======>>>>> Online epoch: #%d, lr=%.10f,Acc=%f,correctnum=%f,allnum=%f,avg_loss=%f  <<<<<======' %(epoch, scheduler.get_lr()[0],acc_avg,acc,total1,loss_avg))
        print("Epoch: " + str(epoch) + " Iter: " + str(i) + " Acc: " + ("%.2f" % acc_avg) +" Classification Loss: " + str(loss_avg))
        logging.info('======>>>>> Online epoch: #%d, lr=%.10f,Acc=%f,correctnum=%f,allnum=%f,avg_loss=%f  <<<<<======' %(epoch, scheduler.get_lr()[0],acc_avg,acc,total1,loss_avg))
        logging.info("Epoch: " + str(epoch) + " Iter: " + str(i) + " Acc: " + ("%.2f" % acc_avg) +" Classification Loss: " + str(loss_avg))
        if ((epoch+1)%1==0 or epoch==opt.nepoch-1):
            # evaluate mode
            torch.cuda.synchronize()
            netR.eval()
            conf_mat = np.zeros([opt.Num_Class, opt.Num_Class])
            conf_mat60 = np.zeros([20, 20])
            acc = 0.0
            loss_sigma = 0.0

            with torch.no_grad():       
                for i, data in enumerate(tqdm(val_loader)):
                    torch.cuda.synchronize()

                    points4DV_T,label,v_name = data
                    # print(v_name)
                    points4DV_T,label = points4DV_T.cuda(),label.cuda()

                    xt, yt = group_points_4DV_T_S(points4DV_T, opt)#(B*F)*4*Cen*K  (B*F)*4*Cen*1
                    
                    xt = xt.type(torch.FloatTensor)
                    yt = yt.type(torch.FloatTensor)

                    prediction = netR(xt,yt)

                    loss = criterion(prediction,label)
                    # print(label,prediction)
                    _, predicted60 = torch.max(prediction.data[:,0:20], 1)
                    _, predicted = torch.max(prediction.data, 1)
                    # print(predicted60.data)
                    loss_sigma += loss.item()

                    for j in range(len(label)):
                        cate_i = label[j].cpu().numpy()
                        pre_i = predicted[j].cpu().numpy()
                        conf_mat[cate_i, pre_i] += 1.0
                        
                        if cate_i<20:
                            pre_i60 = predicted60[j].cpu().numpy()
                            conf_mat60[cate_i, pre_i60] += 1.0
                    # print(conf_mat)
            print('MSR120:{:.2%} MSR60:{:.2%}--correct number {}--all number {}===Average loss:{:.6%}'.format(conf_mat.trace() / conf_mat.sum(),conf_mat60.trace() / conf_mat60.sum(),conf_mat60.trace(),conf_mat60.sum(),loss_sigma/(i+1)/2))
            logging.info('#################{} --epoch{} set Accuracy:{:.2%}--correct number {}--all number {}===Average loss:{}'.format('Valid', epoch, conf_mat.trace() / conf_mat.sum(),conf_mat60.trace(),conf_mat60.sum(), loss_sigma/(i+1)))

        torch.save(netR.module.state_dict(), '%s/pointnet_para_%d.pth' % (opt.save_root_dir, epoch))
if __name__ == '__main__':
    main()

