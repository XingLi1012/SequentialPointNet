import os 
import tqdm
import imageio
import numpy as np
import time
import random
import math
import scipy.io as sio
from PIL import Image#
'''
due to the ntu120 full depth maps data is not avialable, the action proposal is used by skeleton-based action proposal.  
'''


fx = 260.0-20
fy = 240
cx = (20.0+260)/2
cy = (0+240)/2
 

SAMPLE_NUM = 2048
fps_sample_num=512
K = 24  # max frame limit for temporal rank
sample_num_level1=512

save_path = 'C:\\Users\\Administrator\\Desktop\\LX\\paper\\dataset\\Prosessed_dataset\\01_MSR3D\\T'

try:
    os.makedirs(save_path)
except OSError:
    pass



def main():
    data_path='C:\\Users\\Administrator\\Desktop\\LX\\paper\\dataset\\Raw_dataset\\01_MSR3D\\MSR-Action3D(深度.mat)'
    files= os.listdir(data_path)
    for file in files: #
        if(file[-3:]=='mat'):
            if(1==1):#从（a20_s06_e02_sdepth.mat）
                action=int(file[1:3])
                people=int(file[5:7])
                order=int(file[9:11])
                #
                if(action<10):
                    sign_a='A00'
                else:
                    sign_a='A0'

                if(people<10):
                    sign_s='P00'
                else:
                    sign_s='P0'
                
                if(order<10):
                    sign_e='R00'
                else:
                    sign_e='R0'
                #
                filename='S001C001'+sign_s+str(people)+sign_e+str(order)+sign_a+str(action)+'.npy'#文件
                print(filename)
            if(1==1):#
                data=sio.loadmat(data_path+"\\"+file)
                depth=data['depth']
                h, w ,f = depth.shape
            if(1==1):#lx
                ## ------ select a fixed number K of images
                #
                n_frame = f#
                all_sam = np.arange(n_frame)
                if(1==0):#
                    if n_frame    > K:
                        frame_index = random.sample(list(all_sam),K)
                        #frame_index = np.array(frame_index)
                        n_frame = K
                    else:    
                        if n_frame < K/2:
                            frame_index = random.sample(list(all_sam),K-n_frame-n_frame)
                            frame_index.extend(list(all_sam))    
                            frame_index.extend(list(all_sam))    
                            n_frame = K
                        else:
                            frame_index = random.sample(list(all_sam),K-n_frame)
                            frame_index.extend(list(all_sam))
                            n_frame = K
                        frame_index = all_sam.tolist()
                if(1==0):#
                    if n_frame    > K:
                        frame_index=[]
                        for jj in range(K):
                            iii = int(np.random.randint(int(n_frame*jj/K), int(n_frame*(jj+1)/K)), size=1)
                            frame_index.append(iii)
                        n_frame=K
                    else:
                        # frame_index = all_sam.tolist()
                        frame_index = random.sample(list(all_sam),K-n_frame)
                        frame_index.extend(list(all_sam))
                        n_frame = K
                if(1==1):#
                    if n_frame    > K:
                        frame_index=[]
                        for jj in range(K):
                            iii = int((int(n_frame*jj/K)+int(n_frame*(jj+1)/K))/2)
                            frame_index.append(iii)
                        n_frame=K
                    else:
                        # frame_index = all_sam.tolist()
                        frame_index = random.sample(list(all_sam),K-n_frame)
                        frame_index.extend(list(all_sam))
                        n_frame = K
                frame_index.sort()  
                print(f,K,frame_index)            
            if(1==1):#
                all_frame_points_list = []
                depth_Kframe=depth[:,:,frame_index]
                # print(depth_Kframe.shape)
                for i in range(n_frame):
                    # print(depth_Kframe[:,:,i].shape)
                    # if(i==0):
                        # Image.fromarray(depth_Kframe[:,:,i]).show()
                    cloud_im = depth_to_pointcloud(depth_Kframe[:,:,i])
                    # print(cloud_im.shape)#(3, 5356)
                    #lx#print(type(cloud_im))#<class 'numpy.ndarray'>
                    all_frame_points_list.append(cloud_im) #all frame points in 1 list
                    #lx#print(all_frame_points_list.shape)             
            if(1==1):#
                all_frame_3Dpoints_array = np.zeros(shape =[n_frame, SAMPLE_NUM, 3])
                for i in range(n_frame):
                    each_frame_points=all_frame_points_list[i].T#n*3
                    # print('000',each_frame_points.shape)
                    
                    if len(each_frame_points)< SAMPLE_NUM:#lx#
                        if len(each_frame_points)< SAMPLE_NUM/2:
                            if len(each_frame_points)< SAMPLE_NUM/4:
                                # print('01')
                                rand_points_index = np.random.randint(0, each_frame_points.shape[0], size=SAMPLE_NUM-len(each_frame_points)-len(each_frame_points)-len(each_frame_points)-len(each_frame_points))
                                # print(len(rand_points_index))
                                each_frame_points = np.concatenate((each_frame_points, each_frame_points,each_frame_points, each_frame_points,each_frame_points[rand_points_index,:]), axis = 0)
                                # print(each_frame_points.shape)
                            else:
                            
                                # print('11')
                                rand_points_index = np.random.randint(0, each_frame_points.shape[0], size=SAMPLE_NUM-len(each_frame_points)-len(each_frame_points))
                                each_frame_points = np.concatenate((each_frame_points, each_frame_points,each_frame_points[rand_points_index,:]), axis = 0)

                        else:
                            rand_points_index = np.random.randint(0, each_frame_points.shape[0], size=SAMPLE_NUM-len(each_frame_points))
                            each_frame_points = np.concatenate((each_frame_points, each_frame_points[rand_points_index,:]), axis = 0)
                            # print('22')
                    else:#lx#
                        rand_points_index = np.random.randint(0, each_frame_points.shape[0], size=SAMPLE_NUM)
                        each_frame_points = each_frame_points[rand_points_index,:]
                        # print('33')
                    # print(each_frame_points.shape)
                    # all_frame_4Dpoints_array[i]=each_frame_points                 
                    if(1==1):#PFS
                        sampled_idx_l1 = farthest_point_sampling_fast(each_frame_points, sample_num_level1)
                        # print(len(sampled_idx_l1))
                        other_idx = np.setdiff1d(np.arange(SAMPLE_NUM), sampled_idx_l1.ravel())
                        # print(len(other_idx))
                        new_idx = np.concatenate((sampled_idx_l1.ravel(), other_idx))
                        # print(len(new_idx))
                        each_frame_points = each_frame_points[new_idx,:]
                    all_frame_3Dpoints_array[i]=each_frame_points #i:2048*3  
                    # break        
            if(1==1):#
                max_x = all_frame_3Dpoints_array[:,:,0].max()
                max_y = all_frame_3Dpoints_array[:,:,1].max()
                max_z = all_frame_3Dpoints_array[:,:,2].max()
                min_x = all_frame_3Dpoints_array[:,:,0].min()
                min_y = all_frame_3Dpoints_array[:,:,1].min()
                min_z = all_frame_3Dpoints_array[:,:,2].min()
                
                x_len = max_x - min_x
                y_len = max_y - min_y              
                z_len = max_z - min_z
                
                x_center = (max_x + min_x)/2
                y_center = (max_y + min_y)/2
                z_center = (max_z + min_z)/2
                
                all_frame_3Dpoints_array[:,:,0]=(all_frame_3Dpoints_array[:,:,0]-x_center)/y_len
                all_frame_3Dpoints_array[:,:,1]=(all_frame_3Dpoints_array[:,:,1]-y_center)/y_len
                all_frame_3Dpoints_array[:,:,2]=(all_frame_3Dpoints_array[:,:,2]-z_center)/y_len

                save_npy(all_frame_3Dpoints_array, filename)
                # print(all_frame_3Dpoints_array.shape)
                # print(all_frame_3Dpoints_array[0])
                # print(all_frame_4Dpoints_array[1])
                # break
def save_npy(data, filename):
    file = os.path.join(save_path, filename)
    if not os.path.isfile(file):
        np.save(file, data)

def farthest_point_sampling_fast(pc, sample_num):
    pc_num = pc.shape[0]
    #lx#print(pc_num)
    sample_idx = np.zeros(shape = [sample_num,1], dtype = np.int32)
    sample_idx[0] = np.random.randint(0,pc_num)
    #lx#print(sample_idx.shape)
    #lx#print(sample_idx[0])
    #lx#print(pc[sample_idx[0],:])
    cur_sample = np.tile(pc[sample_idx[0],:], (pc_num,1))
    #lx#print(cur_sample.shape)#lx#(2048, 3)
    diff = pc-cur_sample
    #lx#print(diff.shape)#lx#(2048, 3)

    min_dist = (diff*diff).sum(axis = 1)#
    
    #lx#print(min_dist.shape)#lx#(2048,)
    #lx#print(min_dist.reshape(pc_num,1).shape)#lx#(2048,1)
    for cur_sample_idx in range(1,sample_num):#lx#sample_num=512
        ## find the farthest point

        sample_idx[cur_sample_idx] = np.argmax(min_dist)
        # print(sample_idx[cur_sample_idx])
        if cur_sample_idx < sample_num-1:
            diff = pc - np.tile(pc[sample_idx[cur_sample_idx],:], (pc_num,1))
            min_dist = np.concatenate((min_dist.reshape(pc_num,1), (diff*diff).sum(axis = 1).reshape(pc_num,1)), axis = 1).min(axis = 1)  ##?
    #print(min_dist)
    return sample_idx#

def load_depth_from_img(depth_path):
    depth_im = imageio.imread(depth_path) #im is a numpy array
    return depth_im

def depth_to_pointcloud(depth_im):#
    # fx = 2
    # fy = 1
    # cx = 0
    # cy = 0
    #lx#例子输入np.array([[0,2,3],[6,5,3]])
    rows,cols = depth_im.shape#lx#(424, 512)
    #lx#print(depth_im.shape)#lx#(2, 3)
    xx,yy = np.meshgrid(range(0,cols), range(0,rows))
    #lx#print(xx)#lx#[[0 1 2],[0 1 2]]
    #lx#print(yy)#lx#[[0 0 0],[1 1 1]]
    valid = depth_im > 0
    #lx#print(valid)#lx#[[False  True  True],[ True  True  True]]
    xx = xx[valid]#lx#
    yy = yy[valid]#lx#
    #lx#print(xx)#lx#[1 2 0 1 2]
    #lx#print(yy)#lx#[0 0 1 1 1]
    depth_im = depth_im[valid]
    #lx#print(depth_im)#lx#[2 3 6 5 3]
    X = (xx - cx) * depth_im / fx#
    Y = (yy - cy) * depth_im / fy#
    #lx#print(X)#lx#[ -6.   -6.  -24.  -15.   -6. ]
    #lx#print(Y)#lx#[ -5.   -7.5 -12.  -10.   -6. ]
    #lx#print((2-4)*8/1)#lx#-6.0
    Z = depth_im
    points3d = np.array([X.flatten(), Y.flatten(), Z.flatten()])
    #lx#print(points3d)#lx#
    #lx#[[ -6.   -6.  -24.  -15.   -6. ]
    #lx# [ -5.   -7.5 -12.  -10.   -6. ]
    #lx# [  2.    3.    6.    5.    3. ]]
    return points3d

if __name__ == '__main__':
    main()

