from PIL import Image
from keras.models import model_from_json
import os
import utils
import glob
from keras_contrib.layers import InstanceNormalization
import numpy as np
from model import *

# set filepath
datasets=["DRIVE","STARE"]
fundus_dir="../data/{}/test/images/"
mask_dir="../data/{}/test/mask/"
out_dir="../inference_outputs/{}/{}"

model_name = 'atrous_attention_unet'
stare_num =10
drive_num =10
f_weights="../model/{}/{}/g_{}.h5"

for dataset in datasets:
    img_size = (640, 640) if dataset == 'DRIVE' else (720, 720)  # (h,w)  [original img size => DRIVE : (584, 565), STARE : (605,700) ]
    n_filters_g = 32

    if model_name == 'atrous_attention_unet':
        model = generator(img_size, n_filters_g)
    elif model_name == 'unet':
        model = unet(img_size, n_filters_g)
    elif model_name == 'attention_unet':
        model = attention_generator(img_size, n_filters_g)
    elif model_name == 'r2unet':
        model = r2_generator(img_size, n_filters_g)
    elif model_name == 'atrous_unet':
        model = atrous_unet(img_size, n_filters_g)

    # make directory
    if not os.path.isdir(out_dir.format(dataset,model_name)):
        os.makedirs(out_dir.format(dataset, model_name), exist_ok=True)
    
    # load the model and weights
    if dataset == 'STARE':
        model.load_weights(f_weights.format(dataset, model_name, stare_num))
    elif dataset == 'DRIVE':
        model.load_weights(f_weights.format(dataset, model_name, drive_num))
    
    # iterate all images
    img_size=(640,640) if dataset=="DRIVE" else (720,720)
    ori_shape=(1,584,565) if dataset=="DRIVE" else (1,605,700)  # batchsize=1
    fundus_files=utils.all_files_under(fundus_dir.format(dataset))
    mask_files=utils.all_files_under(mask_dir.format(dataset))
    for index,fundus_file in enumerate(fundus_files):
        print("processing {}...".format(fundus_file))
        # load imgs
        img=utils.imagefiles2arrs([fundus_file])
        mask=utils.imagefiles2arrs([mask_files[index]])
        
        # z score with mean, std (batchsize=1)
        mean=np.mean(img[0,...][mask[0,...] == 255.0],axis=0)
        std=np.std(img[0,...][mask[0,...] == 255.0],axis=0)
        img[0,...]=(img[0,...]-mean)/std
        
        # run inference
        padded_img=utils.pad_imgs(img, img_size)
        vessel_img=model.predict(padded_img,batch_size=1)*255
        cropped_vessel=utils.crop_to_original(vessel_img[...,0], ori_shape)
        final_result=utils.remain_in_mask(cropped_vessel[0,...], mask[0,...])
        png_file = os.path.splitext(fundus_file)[0] + '.png'
        Image.fromarray(final_result.astype(np.uint8)).save(os.path.join(out_dir.format(dataset, model_name),os.path.basename(png_file)))
