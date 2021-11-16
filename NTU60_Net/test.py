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

    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 3,  help='number of input point features')
    parser.add_argument('--temperal_num', type=int, default = 3,  help='number of input point features')
    parser.add_argument('--pooling', type=str, default='concatenation', help='how to aggregate temporal split features: vlad | concatenation | bilinear')
    parser.add_argument('--dataset', type=str, default='ntu60', help='how to aggregate temporal split features: ntu120 | ntu60')

    parser.add_argument('--weight_decay', type=float, default=0.0008, help='weight decay (SGD only)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')

    parser.add_argument('--root_path', type=str, default='C:\\Users\\Administrator\\Desktop\\LX\\paper\\dataset\\Prosessed_dataset\\01_3DV-Action-master(base-run-version)(FPS512)(单时序流4维特征)',  help='preprocess folder')
    #parser.add_argument('--depth_path', type=str, default='C:\\Users\\Administrator\\Desktop\\LX\\paper\\dataset\\Prosessed_dataset\\3DV-Action-master\\ntu60dataset\\',  help='raw_depth_png')
    ##############################
    # parser.add_argument('--save_root_dir', type=str, default='C:\\Users\\Administrator\\Desktop\\LX\\paper\\code\\3DV-Action-master\\models\\ntu60\\xsub',  help='output folder')
    parser.add_argument('--save_root_dir', type=str, default='C:\\Users\\Administrator\\Desktop\\LX\\paper\\code\\Models_Parameter\\03_3DV-Action-master(base-run-version)(FPS512-64-K32)(2层局部+时序池化3层简单)(t=6_stride=2_KC=64)(单时序流4维特征)(单流读取)\\models\\ntu60\\xview',  help='output folder')
    parser.add_argument('--model', type=str, default = '',  help='model name for training resume')
    parser.add_argument('--optimizer', type=str, default = '',  help='optimizer name for training resume')
    
    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python train.py

    ########
    parser.add_argument('--Seg_size', type=int, default =1,  help='number of frame in seg')
    parser.add_argument('--stride', type=int, default = 1,  help='stride of seg')
    parser.add_argument('--all_framenum', type=int, default = 20,  help='number of action frame')
    parser.add_argument('--framenum', type=int, default = 20,  help='number of action frame')
    parser.add_argument('--EACH_FRAME_SAMPLE_NUM', type=int, default = 512,  help='number of sample points in each frame')
    parser.add_argument('--T_knn_K', type=int, default = 48,  help='K for knn search of temperal stream')
    parser.add_argument('--T_knn_K2', type=int, default = 16,  help='K for knn search of temperal stream')
    parser.add_argument('--T_sample_num_level1', type=int, default = 128,  help='number of first layer groups')
    parser.add_argument('--T_sample_num_level2', type=int, default = 32,  help='number of first layer groups')
    parser.add_argument('--T_ball_radius', type=float, default=0.2, help='square of radius for ball query of temperal stream')
    
    parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')

    parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
    parser.add_argument('--SAMPLE_NUM', type=int, default = 2048,  help='number of sample points')

    parser.add_argument('--Num_Class', type=int, default = 60,  help='number of outputs')
    parser.add_argument('--knn_K', type=int, default = 64,  help='K for knn search')
    parser.add_argument('--sample_num_level1', type=int, default = 512,  help='number of first layer groups')
    parser.add_argument('--sample_num_level2', type=int, default = 128,  help='number of second layer groups')
    parser.add_argument('--ball_radius', type=float, default=0.06, help='square of radius for ball query in level 1')#0.025 -> 0.05 for detph
    parser.add_argument('--ball_radius2', type=float, default=0.1, help='square of radius for ball query in level 2')# 0.08 -> 0.01 for depth

    opt = parser.parse_args()
    print (opt)
    # logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=os.path.join(opt.save_root_dir, '(单3dp)(去2)(2局+3简运)(位编)(cs2t1)(多池1212-66)(中2)T11F20-51212832k4816test.log'), level=logging.INFO)
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

    ################2##############
    data_val = NTU_RGBD(root_path = opt.root_path, opt=opt,
        DATA_CROSS_VIEW = True,
        full_train = False,
        validation = False,
        test = True,
        Transform = False
        )
    val_loader = DataLoader(dataset = data_val, batch_size = 8,num_workers = 8)

    #net =

    netR = PointNet_Plus(opt)
    #################3############
    netR.load_state_dict(torch.load("C:\\Users\\Administrator\\Desktop\\pointnet_para_82.pth"))

    netR = torch.nn.DataParallel(netR).cuda()
    netR.cuda()
    print(netR)

    
    # evaluate mode
    torch.cuda.synchronize()
    netR.eval()
    conf_mat = np.zeros([opt.Num_Class, opt.Num_Class])
    conf_mat60 = np.zeros([60, 60])
    acc = 0.0
    loss_sigma = 0.0

    with torch.no_grad(): 
        for i, data in enumerate(tqdm(val_loader)):
            #print(i)
            torch.cuda.synchronize()
            group_time_start = time.time()
            points4DV_T,label,vid_name = data
            points4DV_T,label = points4DV_T.cuda(),label.cuda()
            # print('points4DV_T:',points4DV_T.shape)
            xt, yt = group_points_4DV_T_S(points4DV_T, opt)#B*F*4*Cen*K  B*F*4*Cen*1
            # print('xt:',xt.shape)
            xt = xt.type(torch.FloatTensor)
            yt = yt.type(torch.FloatTensor)
            forward_time_start= time.time()
            prediction = netR(xt,yt)
            forward_time_end = time.time()

         
            print('forward time:',forward_time_end-forward_time_start)
            _, predicted60 = torch.max(prediction.data[:,0:60], 1)
            _, predicted = torch.max(prediction.data, 1)
            #print(prediction.data)
            
            for j in range(len(label)):
                cate_i = label[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                if pre_i != cate_i:
                    logging.info('Video Name:{} -- correct label {} predicted to {}'.format(vid_name[j],cate_i,pre_i))
                conf_mat[cate_i, pre_i] += 1.0
                if cate_i<60:
                    pre_i60 = predicted60[j].cpu().numpy()
                    conf_mat60[cate_i, pre_i60] += 1.0

    print('NTU120:{:.2%} NTU60:{:.2%}--correct number {}--all number {}===Average loss:{:.6%}'.format(conf_mat.trace() / conf_mat.sum(),conf_mat60.trace() / conf_mat60.sum(),conf_mat60.trace(),conf_mat60.sum(),loss_sigma/(i+1)/2))
    # logging.info('{} --nepoch{} set Accuracy:{:.2%}--correct number {}--all number {}===Average loss:{}'.format('Valid', opt.nepoch, conf_mat.trace() / conf_mat.sum(),conf_mat60.trace(),conf_mat60.sum(), loss_sigma/(i+1)))

        #torch.save(netR.module.state_dict(), '%s/pointnet_para_%d.pth' % (opt.save_root_dir, epoch))
if __name__ == '__main__':
    main()

