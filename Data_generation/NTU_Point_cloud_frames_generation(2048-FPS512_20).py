import os 
import tqdm
import imageio
import numpy as np
import time
import random
import math
import scipy.io as sio
from PIL import Image#

fx = 365.481
fy = 365.481
cx = 257.346
cy = 210.347
 

SAMPLE_NUM = 2048
fps_sample_num=512
K = 20  # max frame limit for temporal rank

save_path = 'C:\\Users\\Administrator\\Desktop\\LX\\paper\\dataset\\Prosessed_dataset\\4DV(去地面)\\T120'

try:
    os.makedirs(save_path)
except OSError:
    pass

def main():
    data_path = 'C:\\Users\\Administrator\\Desktop\\LX\\paper\\dataset\\Raw_dataset\\ntu60dataset'
    sub_Files = os.listdir(data_path)
    sub_Files.sort()
    index17=0

    for s_fileName in sub_Files:#lx#Traverse 17 settings folders
        index17=index17+1
        if(index17>0):
            videoPath = os.path.join(data_path, s_fileName, 'nturgb+d_depth_masked')
            if os.path.isdir(videoPath):#lx#Determine whether it is a directory
                print(s_fileName)
                video_Files = os.listdir(videoPath)
                #print(video_Files)
                video_Files.sort()
                video_index=0
                video_num=len(video_Files)
                print(time.time())
                for video_FileName in video_Files:#lx#Traverse the sample folder
                    video_index=video_index+1
                    # if(video_num*3/4<video_index):
                    # if(video_num/2<video_index<=video_num*3/4):
                    # if(video_num/4<video_index<=video_num/2):
                    if(video_index<=video_num/4):
                        print(video_FileName)

                        filename = video_FileName +'.npy'
                        file = os.path.join(save_path, filename)
                        if os.path.isfile(file):
                            continue
                        pngPath = os.path.join(videoPath,video_FileName)
                        imgNames = os.listdir(pngPath)
                        imgNames.sort()
                        ## ------ select a fixed number K of images
                        if(1==1):#lxMake random 20-frame subscripts
                            ## ------ select a fixed number K of images
                            #Make an index list frame_index, randomly sample images larger than 20 frames, and randomly sample to 20 frames smaller than 60 frames.
                            n_frame = len(imgNames)#lx#
                            all_sam = np.arange(n_frame)
                            if(1==0):#Randomly sampled frames
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
                            if(1==0):#Randomly sample frames at equal intervals
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
                            if(1==1):#Sampling frames at equal intervals
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
                            # print(frame_index)

                        ### ------convert the depth sequence to points data
                        # print(n_frame,K)
                        if(1==1):#Read the point cloud of each frame and sample
                            all_frame_3Dpoints_array = np.zeros(shape =[n_frame, fps_sample_num, 3])
                            i=0
                            for i_frame in frame_index:#lx#Traverse frames
                                depthName = imgNames[i_frame]
                                img_path = os.path.join(pngPath,depthName)
                                depth_im = load_depth_from_img(img_path)
                                #lx#print(depth_im.shape)#(424, 512)
                                each_frame_points = depth_to_pointcloud(depth_im).T
                                #lx#print(cloud_im.shape)#(3, 5356)
                                #lx#print(type(cloud_im))#<class 'numpy.ndarray'>
                                if(1==1):#Randomly sample 2048 points
                                    if len(each_frame_points)< SAMPLE_NUM:#lx#If the number of points is less than 2048, it will be repeated randomly to 2048
                                        if len(each_frame_points)< SAMPLE_NUM/2:
                                            if len(each_frame_points)< SAMPLE_NUM/4:
                                                # print('01')
                                                rand_points_index = np.random.randint(0, each_frame_points.shape[0], size=SAMPLE_NUM-len(each_frame_points)-len(each_frame_points)-len(each_frame_points)-len(each_frame_points))
                                                # print(len(rand_points_index))
                                                each_frame_points = np.concatenate((each_frame_points, each_frame_points,each_frame_points, each_frame_points,each_frame_points[rand_points_index,:]), axis = 0)
                                                # print(each_frame_points.shape)
                                            else:
                                                rand_points_index = np.random.randint(0, each_frame_points.shape[0], size=SAMPLE_NUM-len(each_frame_points)-len(each_frame_points))
                                                each_frame_points = np.concatenate((each_frame_points, each_frame_points,each_frame_points[rand_points_index,:]), axis = 0)

                                        else:
                                            rand_points_index = np.random.randint(0, each_frame_points.shape[0], size=SAMPLE_NUM-len(each_frame_points))
                                            each_frame_points = np.concatenate((each_frame_points, each_frame_points[rand_points_index,:]), axis = 0)
                                            # print('22')
                                    else:#lx#If it exceeds 2048, randomly sample 2048 points
                                        rand_points_index = np.random.randint(0, each_frame_points.shape[0], size=SAMPLE_NUM)
                                        each_frame_points = each_frame_points[rand_points_index,:]
                                        # print('33')
                                    # print(each_frame_points.shape)
                                    # all_frame_4Dpoints_array[i]=each_frame_points                 
                                if(1==1):#PFS
                                    sampled_idx_l1 = farthest_point_sampling_fast(each_frame_points, fps_sample_num)
                                    # print(len(sampled_idx_l1))
                                    other_idx = np.setdiff1d(np.arange(SAMPLE_NUM), sampled_idx_l1.ravel())
                                    # print(len(other_idx))
                                    new_idx = np.concatenate((sampled_idx_l1.ravel(), other_idx))
                                    # print(len(new_idx))
                                    each_frame_points = each_frame_points[new_idx,:]
                                all_frame_3Dpoints_array[i]=each_frame_points[:fps_sample_num,:] #i:512*3
                                i=i+1
                        if(1==1):#Normalized
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
                            if(video_index==100):
                                print(time.time())
                            # save_npy(all_frame_3Dpoints_array, filename)


 
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
    #lx#print((diff*diff).shape)#lx#(2048, 3)(diff*diff)
    min_dist = (diff*diff).sum(axis = 1)#lx#
    
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
    return sample_idx#lx#

def load_depth_from_img(depth_path):
    depth_im = imageio.imread(depth_path) #im is a numpy array
    return depth_im

def depth_to_pointcloud(depth_im):#lx#
    # fx = 2
    # fy = 1
    # cx = 0
    # cy = 0
    #lx#np.array([[0,2,3],[6,5,3]])
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
    X = (xx - cx) * depth_im / fx#LX#
    Y = (yy - cy) * depth_im / fy#LX#
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

