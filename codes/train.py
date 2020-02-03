import numpy as np
import time
from model import *
import utils
import os
from PIL import Image
import argparse
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.utils import generic_utils

# arrange arguments
parser=argparse.ArgumentParser()
parser.add_argument(
    '--ratio_gan2seg',
    type=int,
    help="ratio of gan loss to seg loss",
    required=True
    )
parser.add_argument(
    '--gpu_index',
    type=str,
    help="gpu index",
    required=True
    )
parser.add_argument(
    '--batch_size',
    type=int,
    help="batch size",
    required=True
    )
parser.add_argument(
    '--dataset',
    type=str,
    help="dataset name",
    required=True
    )
FLAGS,_= parser.parse_known_args()

# training settings
os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu_index
n_rounds=10
batch_size=FLAGS.batch_size
n_filters_d=32
n_filters_g=32
val_ratio=0.1
init_lr=3e-4
schedules={'lr_decay':{},  # learning rate and step have the same decay schedule (not necessarily the values)
           'step_decay':{}}
alpha_recip=1./FLAGS.ratio_gan2seg if FLAGS.ratio_gan2seg>0 else 0
rounds_for_evaluation=range(n_rounds)

# set dataset
print("setting dataset...")
dataset=FLAGS.dataset
img_size= (640,640) if dataset=='DRIVE' else (720,720) # (h,w)  [original img size => DRIVE : (584, 565), STARE : (605,700) ]

model_name = 'atrous_attention_unet'

img_out_dir="../training_process/{}/{}".format(FLAGS.dataset,model_name)
model_out_dir="../model/{}/{}".format(FLAGS.dataset,model_name)
auc_out_dir="../auc/{}/{}".format(FLAGS.dataset,model_name)
train_dir="../data/{}/training/".format(dataset)
test_dir="../data/{}/test/".format(dataset)
if not os.path.isdir(img_out_dir):
    os.makedirs(img_out_dir, exist_ok=True)
if not os.path.isdir(model_out_dir):
    os.makedirs(model_out_dir, exist_ok=True)
if not os.path.isdir(auc_out_dir):
    os.makedirs(auc_out_dir, exist_ok=True)
print("finished setting dataset")

# set training and validation dataset
print("setting training dataset...")
train_imgs, train_vessels =utils.get_imgs(train_dir, augmentation=True, img_size=img_size, dataset=dataset)
train_vessels=np.expand_dims(train_vessels, axis=3)
n_all_imgs=train_imgs.shape[0]
n_train_imgs=int((1-val_ratio)*n_all_imgs)
train_indices=np.random.choice(n_all_imgs,n_train_imgs,replace=False)
train_batch_fetcher=utils.TrainBatchFetcher(train_imgs[train_indices,...], train_vessels[train_indices,...], batch_size)
print("finish setting training dataset")
print("setting validation dataset...")
val_imgs, val_vessels=train_imgs[np.delete(range(n_all_imgs),train_indices),...], train_vessels[np.delete(range(n_all_imgs),train_indices),...]
print("finish setting validation dataset")
# set test dataset
print("setting test dataset...")
test_imgs, test_vessels, test_masks=utils.get_imgs(test_dir, augmentation=False, img_size=img_size, dataset=dataset, mask=True)
print("finish setting test dataset")

# log_dir = '../log/{}-{}/'.format(dataset,FLAGS.discriminator)
# if not os.path.isdir(log_dir):
#     os.makedirs(log_dir)
#
# callback = TensorBoard(log_dir)

# create networks
if model_name == 'atrous_attention_unet':
    g = generator(img_size, n_filters_g)
elif model_name == 'unet':
    g = unet(img_size, n_filters_g)
elif model_name == 'r2unet':
    g = r2_generator(img_size, n_filters_g)
elif model_name =='attention_unet':
    g = attention_generator(img_size, n_filters_g)
elif model_name == 'atrous_unet':
    g = atrous_unet(img_size, n_filters_g)

g.summary()
d, d_out_shape = discriminator(img_size, n_filters_d,init_lr)

d.summary()

gan=GAN(g,d,img_size, n_filters_g, n_filters_d,alpha_recip, init_lr)
gan.summary()

# start training
scheduler=utils.Scheduler(n_train_imgs//batch_size, n_train_imgs//batch_size, schedules, init_lr) if alpha_recip>0 else utils.Scheduler(0, n_train_imgs//batch_size, schedules, init_lr)
print("training {} images :".format(n_train_imgs))

# # write discriminator and generator loss logs
# def write_log(callback, names, logs, batch_no):
#     for name, value in zip(names, logs):
#         summary = tf.Summary()
#         summary_value = summary.value.add()
#         summary_value.simple_value = value
#         summary_value.tag = name
#         callback.writer.add_summary(summary, batch_no)
#         callback.writer.flush()

start = time.time()
for n_round in range(n_rounds):
    # train D
    steps = n_train_imgs//batch_size

    utils.make_trainable(d, True)
    d_progbar = generic_utils.Progbar(n_train_imgs)
    for batch_no in range(steps):
        real_imgs, real_vessels = next(train_batch_fetcher)
        fake_vessels = g.predict(real_imgs,batch_size=batch_size)

        d_x_batch, d_y_batch = utils.input2discriminator(real_imgs, real_vessels, fake_vessels, d_out_shape, train_real=True)
        d_loss = d.train_on_batch(d_x_batch, d_y_batch)
        # write_log(callback, ['d_loss'], d_loss, batch_no)
        d_progbar.add(batch_size, values=[("Loss_D", d_loss[0]), ("Accuracy_D", d_loss[1])])

    # train G (freeze discriminator)
    utils.make_trainable(d, False)
    g_progbar = generic_utils.Progbar(n_train_imgs)
    for batch_no in range(steps):
        real_imgs, real_vessels = next(train_batch_fetcher)

        g_x_batch, g_y_batch=utils.input2gan(real_imgs, real_vessels, d_out_shape, train_real=True)
        g_loss = gan.train_on_batch(g_x_batch, g_y_batch)
        # write_log(callback, ['g_loss'],g1_loss, batch_no)

        g_progbar.add(batch_size, values=[("Loss_G", g_loss[0]), ("Accuracy_G", g_loss[1])])

    # evaluate on validation set
    if n_round in rounds_for_evaluation:
        # D
        fake_val_vessels = g.predict(val_imgs,batch_size=batch_size)
        d_x_test, d_y_test=utils.input2discriminator(val_imgs, val_vessels, fake_val_vessels, d_out_shape, train_real=True)
        loss, acc=d.evaluate(d_x_test,d_y_test, batch_size=batch_size, verbose=0)
        utils.print_metrics(n_round+1, loss=loss, acc=acc, type='D')
        # G
        gan_x_test, gan_y_test=utils.input2gan(val_imgs, val_vessels, d_out_shape, train_real=True)
        loss,acc=gan.evaluate(gan_x_test,gan_y_test, batch_size=batch_size, verbose=0)
        utils.print_metrics(n_round+1, acc=acc, loss=loss, type='GAN')

        # save the model and weights with the best validation loss
        with open(os.path.join(model_out_dir,"g_{}.json".format(n_round+1)),'w') as f:
            f.write(g.to_json())
        g.save_weights(os.path.join(model_out_dir,"g_{}.h5".format(n_round+1)))

    # update step sizes, learning rates
    scheduler.update_steps(n_round)
    K.set_value(d.optimizer.lr, scheduler.get_lr())
    K.set_value(gan.optimizer.lr, scheduler.get_lr())
    
    # evaluate on test images
    if n_round in rounds_for_evaluation:    
        generated=g.predict(test_imgs,batch_size=batch_size)
        generated=np.squeeze(generated, axis=3)
        vessels_in_mask, generated_in_mask = utils.pixel_values_in_mask(test_vessels, generated , test_masks)
        auc_roc=utils.AUC_ROC(vessels_in_mask,generated_in_mask,os.path.join(auc_out_dir,"auc_roc_{}.npy".format(n_round+1)))
        auc_pr=utils.AUC_PR(vessels_in_mask, generated_in_mask,os.path.join(auc_out_dir,"auc_pr_{}.npy".format(n_round+1

                                                                                                               )))
        utils.print_metrics(n_round+1, auc_pr=auc_pr, auc_roc=auc_roc, type='TESTING')
         
        # print test images
        img_dir = os.path.join(img_out_dir, str(n_round+1))
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)

        segmented_vessel=utils.remain_in_mask(generated, test_masks)
        for index in range(segmented_vessel.shape[0]):
            Image.fromarray((segmented_vessel[index,:,:]*255).astype(np.uint8)).save(os.path.join(img_dir,"{:02}_segmented.png".format(index+1)))

total_time = time.time() - start
print('Total Training Time: {}'.format(time.strftime('%Hh %Mm %Ss', time.gmtime(total_time))))