import torch
import math

import torch.nn as nn
import torch.nn.functional as F

from utils import group_points_4DV_T_S,group_points_4DV_T_S2
from channelattention import ChannelAttention,ChannelAttention0
from positionencoding import get_positional_encoding
nstates_plus_1 = [64,64,128]
nstates_plus_2 = [128,128,256]
nstates_plus_3 = [256,512,1024,1024,256]

S_nstates_plus_1 = [64,64,128]
S_nstates_plus_2 = [128,128,256]
T_nstates_plus_2 = [256,512,1024]
T_nstates_plus_3 = [1024]
vlad_dim_out = 128*8

dim_out=1024

class PointNet_Plus(nn.Module):
    def __init__(self,opt,num_clusters=8,gost=1,dim=128,normalize_input=True):
        super(PointNet_Plus, self).__init__()
        self.temperal_num = opt.temperal_num
        self.knn_K = opt.knn_K
        self.ball_radius2 = opt.ball_radius2
        self.sample_num_level1 = opt.sample_num_level1
        self.sample_num_level2 = opt.sample_num_level2
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM # x,y,x,c : 4
        self.num_outputs = opt.Num_Class
        ####SAMPLE_NUM
        self.Seg_size = opt.Seg_size
        self.stride=opt.stride
        self.EACH_FRAME_SAMPLE_NUM=opt.EACH_FRAME_SAMPLE_NUM
        self.T_knn_K = opt.T_knn_K
        self.T_knn_K2= opt.T_knn_K2
        self.T_sample_num_level1 = opt.T_sample_num_level1
        self.T_sample_num_level2 = opt.T_sample_num_level2
        self.framenum=opt.framenum
        self.T_group_num=int((self.framenum-self.Seg_size)/self.stride)+1

        self.opt=opt
        self.dim=dim

        self.normalize_input=normalize_input

        self.pooling = opt.pooling
        #self._init_params()


        self.netR_T_S1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM+1, S_nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(S_nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(S_nstates_plus_1[0], S_nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(S_nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(S_nstates_plus_1[1], S_nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(S_nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1,self.T_knn_K),stride=1)
            )
        self.ca_S2 = ChannelAttention(self.INPUT_FEATURE_NUM+1+S_nstates_plus_1[2])
        self.netR_T_S2 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM+1+S_nstates_plus_1[2], S_nstates_plus_2[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(S_nstates_plus_2[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(S_nstates_plus_2[0], S_nstates_plus_2[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(S_nstates_plus_2[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(S_nstates_plus_2[1], S_nstates_plus_2[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(S_nstates_plus_2[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1,self.T_knn_K2),stride=1)
            )
        self.ca_T1 = ChannelAttention(self.INPUT_FEATURE_NUM+S_nstates_plus_2[2])
        self.net4DV_T1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K(B*10*28*2048)
            nn.Conv2d(self.INPUT_FEATURE_NUM+S_nstates_plus_2[2], T_nstates_plus_2[0], kernel_size=(1, 1)),#10->64
            nn.BatchNorm2d(T_nstates_plus_2[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(T_nstates_plus_2[0], T_nstates_plus_2[1], kernel_size=(1, 1)),#64->64
            nn.BatchNorm2d(T_nstates_plus_2[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(T_nstates_plus_2[1], T_nstates_plus_2[2], kernel_size=(1, 1)),#64->128
            nn.BatchNorm2d(T_nstates_plus_2[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            # nn.Conv2d(T_nstates_plus_2[2], T_nstates_plus_2[3], kernel_size=(1, 1)),#64->128
            # nn.BatchNorm2d(T_nstates_plus_2[3]),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d((1,self.T_sample_num_level2*self.Seg_size),stride=1)#1*（t*512）#B*C*G*1
            )
        self.net4DV_T2 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(T_nstates_plus_2[2], T_nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(T_nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            # nn.Conv2d(T_nstates_plus_3[0], T_nstates_plus_3[1], kernel_size=(1, 1)),
            # nn.BatchNorm2d(T_nstates_plus_3[1]),
            # nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            # nn.Conv2d(T_nstates_plus_3[1], T_nstates_plus_3[2], kernel_size=(1, 1)),
            # nn.BatchNorm2d(T_nstates_plus_3[2]),
            # nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            # nn.MaxPool2d((self.T_group_num,1),stride=1),
            # B*1024*1*1
        )

        KerStr=[(24,24),(12,12)]
        self.maxpoolings = nn.ModuleList([nn.MaxPool2d((K[0],1 ),(K[1],1)) for K in KerStr])
        self.PE=get_positional_encoding(self.framenum,T_nstates_plus_2[2])

        self.netR_FC = nn.Sequential(
            # B*1024
            #nn.Linear(nstates_plus_3[2], nstates_plus_3[3]),
            #nn.BatchNorm1d(nstates_plus_3[3]),
            #nn.ReLU(inplace=True),
            # B*1024
            nn.Linear(dim_out*4+256, nstates_plus_3[4]),
            nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_3[4], self.num_outputs),
            nn.BatchNorm1d(self.num_outputs),
            nn.ReLU(inplace=True),
            # B*num_outputs
        )

    def forward(self, xt, yt):
    
        
        B,f,d,N,k = xt.shape#B*F*4*Cen*K
        # print('xt:',xt.shape)
        # print('yt:',yt.shape)
        yt=yt.view(B*f,yt.size(2), self.opt.T_sample_num_level1, 1)#(B*F)*4*Cen*1
        xt=xt.view(B*f,d, self.opt.T_sample_num_level1, k)#(B*F)*4+1*Cen*K
        # print('xtt:',xt.shape)
        # xt = self.ca_S1(xt) * xt
        xt = self.netR_T_S1(xt)#(B*F)*128*Cen*1
        # print('xttt:',xt.shape)
        xt = torch.cat((yt, xt),1).squeeze(-1)#(B*F)*(4+128)*Cen
        xt=xt.view(B,f,xt.size(1), self.opt.T_sample_num_level1).transpose(2,3)#(B*F)*(4+128)*Cen->B*F*Cen1*(4+128)
        # print('xtttt:',xt.shape)
        S_inputs_level2,inputs_level1_center_s2 =group_points_4DV_T_S2(xt,self.opt)##B*F*5+128*Cen2*K2   B*F*4*Cen2*1
        # print('S_inputs_level2:',S_inputs_level2.shape)
        B2,f2,d2,N2,k2 = S_inputs_level2.shape#B*F*4*Cen*K
        inputs_level1_center_s2=inputs_level1_center_s2.view(B2*f2,inputs_level1_center_s2.size(2), self.opt.T_sample_num_level2, 1)#(B*F)*4*C2en*1
        S_inputs_level2=S_inputs_level2.view(B2*f2,d2, self.opt.T_sample_num_level2, k2)#(B*F)*5+128*C2en*K2
        S_inputs_level2 = self.ca_S2(S_inputs_level2) * S_inputs_level2
        xt = self.netR_T_S2(S_inputs_level2)#(B*F)*128*Cen2*1
        
        ###res s2
        xt_resS2=xt.squeeze(-1).view(B,f,xt.size(1), self.opt.T_sample_num_level2).transpose(1,2)#B*256*F*Cen2
        xt_resS2=F.max_pool2d(xt_resS2,kernel_size=(f,self.opt.T_sample_num_level2)).squeeze(-1).squeeze(-1)#B*256
        
        # print('xxt:',xt.shape)
        xt = torch.cat((inputs_level1_center_s2, xt),1).squeeze(-1)#(B*F)*4+128*Cen2
        xt =xt.view(-1,self.framenum,xt.size(1),self.opt.T_sample_num_level2).transpose(2,3)##(B*F)*(4+128)*C2en-》B*F*(4+128)*C2en->B*F*C2en*(4+128)
        
        
        # print(-1,self.framenum,4+128,self.opt.T_sample_num_level1,xt.shape)
        # print('xtttt:',xt.shape)
        T_inputs_level2 =xt.transpose(1,3).transpose(2,3)
        T_inputs_level2 = self.ca_T1(T_inputs_level2) * T_inputs_level2
        xt = self.net4DV_T1(T_inputs_level2)# 
        
        ###resT1
        xt_resT1=F.max_pool2d(xt,kernel_size=(f,1)).squeeze(-1).squeeze(-1)#B*256
        #
        xt=xt.squeeze(-1)+self.PE.transpose(0,1)
        xt=xt.unsqueeze(-1)
        ###
        
        
        # xt = torch.cat((inputs_level1_center_t, xt),1)
        # xt = self.ca_T2(xt) * xt
        xt = self.net4DV_T2(xt)#

        xt = [maxpooling(xt) for maxpooling in self.maxpoolings]#B*(2048)*[G]*1
        xt = torch.cat(xt,2).squeeze(-1)
        # print('xttttt:',xt.shape)
        # print(xt.size(0),xt.size(1)*xt.size(2))
        xt = xt.contiguous().view(xt.size(0),-1)
        
        #######
        xt = torch.cat((xt, xt_resS2,xt_resT1),1)
        # print('xttt:',xt.shape)
        x = self.netR_FC(xt)
        # print('x:',x.shape)
        
        return x



