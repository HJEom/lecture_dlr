import tensorflow as tf
import numpy as np
import cv2   # (h, w, c) , cv2.imread fcn stores image in (h, w, BGR) by default.
import sys
import os
import time
try:
    os.path.exists(sys.argv[1])
except:
    print("usage : $1 /<dataset>/<directory>/<path>/SR_dataset/291/")
    sys.exit(1)
train_img_path = sys.argv[1]

#################################### to save weights and biases
if os.path.isdir("./results/params/") == False:
    os.makedirs("./results/params/")
params_path = './results/params/'

def var_init(name_, shape_):
    return tf.get_variable(name_, shape=shape_, initializer=tf.contrib.layers.xavier_initializer_conv2d())

def conv2d(in_, w_, b_):
    return tf.nn.conv2d(in_,w_,strides=[1,1,1,1], padding='SAME') + b_

# cv2.resize library requires dsize that is (width, height, ch)
def img_resize(image, downsize, upsize):
    image = cv2.resize(image, dsize=downsize, interpolation=cv2.INTER_CUBIC);
    image = cv2.resize(image, dsize=upsize, interpolation=cv2.INTER_CUBIC);
    return image.reshape(image.shape[0], image.shape[1],1)

def get_all_dataset(image_path, image_list):
    high_images = np.empty(len(image_list), dtype=object)
    low_images = np.empty(len(image_list), dtype=object)
    print("\ngetting all dataset for train and test...")
    for i in range(len(image_list)):
        image = cv2.cvtColor(cv2.imread(image_path+'/'+image_list[i]),cv2.COLOR_BGR2GRAY)
        high_images[i] = image.reshape(image.shape[0], image.shape[1],1)
        low_images[i] = img_resize(image, (int(image.shape[1]/2),int(image.shape[0]/2)),(image.shape[1],image.shape[0]))
    return high_images, low_images

def random_crop(high_images, low_images, mini_batch_size, crop_size, train, train_rate):
    crop_high_img = np.empty((mini_batch_size,crop_size,crop_size,1),dtype=np.uint8)
    crop_low_img= np.empty((mini_batch_size,crop_size,crop_size,1),dtype=np.uint8)
    for i in range(mini_batch_size):
        if train == 1:
            mini_batch_number = np.random.random_integers(0,int(high_images.shape[0]*train_rate)-1)
        else:
            mini_batch_number = np.random.random_integers((int(high_images.shape[0]*train_rate)), (int(high_images.shape[0]-1)))
        h_ = np.random.random_integers(0, high_images[mini_batch_number].shape[0]-crop_size)
        w_ = np.random.random_integers(0, high_images[mini_batch_number].shape[1]-crop_size)
        crop_high_img[i] = high_images[mini_batch_number][h_:h_+crop_size, w_:w_+crop_size, :]
        crop_low_img[i] = low_images[mini_batch_number][h_:h_+crop_size, w_:w_+crop_size, :]
    return crop_high_img, crop_low_img, crop_high_img/255.0, crop_low_img/255.0

iteration = 1
lr = 0.0001    # 10^-5
mini_batch_size = 128
crop_size = 32
train_test_rate = 0.85    # The number of images for train is 247 and for test is 44 when train_test_rate is 0.85.
ch=1

#################################### for train
in_img_f = tf.placeholder(dtype=tf.float32, shape=[None,None,None,ch],name="in_img_f")
label_img_f = tf.placeholder(dtype=tf.float32, shape=[None,None,None,ch],name="label_img_f")
#################################### calculate psnr to dtype=uint8
out_img_uint8 = tf.placeholder(dtype=tf.uint8, shape=[None,None,None,ch],name="out_img_uint8")
label_img_uint8 = tf.placeholder(dtype=tf.uint8, shape=[None,None,None,ch],name="label_img_uint8")

#################################### weight and bias
with tf.name_scope("layer1_params"):
    w1 = var_init("w1",[3,3,ch,64])
    b1 = var_init("b1",[64])

with tf.name_scope("layer2_params"):
    w2 = var_init("w2",[3,3,64,64])
    b2 = var_init("b2",[64])

with tf.name_scope("layer3_params"):
    w3 = var_init("w3",[3,3,64,ch])
    b3 = var_init("b3",[ch])

#################################### network architecture
layer1_out = tf.nn.relu(conv2d(in_img_f,w1,b1))
layer2_out = tf.nn.relu(conv2d(layer1_out,w2,b2))
with tf.name_scope("layer3_out"):    # for evaluation
    layer3_out = conv2d(layer2_out,w3,b3)

#################################### cost
with tf.name_scope("cost"):    # for tensorboard
    cost = tf.reduce_sum(pow((layer3_out-label_img_f), 2))/mini_batch_size
    tf.summary.scalar("cost", cost)    # for tensorboard

#################################### psnr
with tf.name_scope("psnr"):    # for tensorboard
    psnr = tf.reduce_mean(tf.image.psnr(out_img_uint8, label_img_uint8, max_val=255))
    tf.summary.scalar("psnr", psnr)    # for tensorboard

#################################### train
train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

#################################### to save parameters.
save_path = params_path + "./train.ckpt"
saver = tf.train.Saver()

              # config is to show which the graph uses gpu or cpu.
with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    merged_summary = tf.summary.merge_all()    # for tensorboard
    writer = tf.summary.FileWriter("./results/logs/SRCNN")    # for tensorboard
    writer.add_graph(sess.graph)    # for tensorboard

    #################################### initialize variables
    sess.run(tf.global_variables_initializer())

    #################################### get dataset for train
    start_time=time.time()
    train_img_list = os.listdir(train_img_path+'/')    # length = 291
    with tf.device('/cpu:0'):
        high_img, low_img = get_all_dataset(train_img_path, train_img_list)    # dimensions (291, h, w, 1)
    print(" %s seconds \n" % (time.time() - start_time))

    print("start train and test...")
    for itr in range(iteration+1):
        #################################### train
        ##### fetch mini-batch data
        train_high_img_uint8, train_low_img_uint8, train_high_img_float32, train_low_img_float32 = random_crop(high_img, low_img, mini_batch_size, crop_size, 1, train_test_rate)

        train_high_img_float32 = train_high_img_float32.astype(np.float32)
        train_low_img_float32 = train_low_img_float32.astype(np.float32)

        ##### train and cost
        train_, cost_ = sess.run([train, cost],
                feed_dict={in_img_f : train_low_img_float32, label_img_f : train_high_img_float32})

        ##### forward pass to calculate psnr
        layer3_f = sess.run(layer3_out,
                feed_dict={in_img_f : train_low_img_float32})
        layer3_uint8 = (layer3_f*255.0).astype(np.uint8)

        ##### calculate psnr
        psnr_, summary_ = sess.run([psnr, merged_summary],
                feed_dict={label_img_uint8 : train_high_img_uint8, out_img_uint8 : layer3_uint8, in_img_f : train_low_img_float32, label_img_f : train_high_img_float32})
        writer.add_summary(summary_, global_step=itr)    # for tensorboard

        ##################################### test
        if itr%100 == 0:
            test_high_img_uint8, test_low_img_uint8, test_high_img_float32, test_low_img_float32 = random_crop(high_img, low_img, mini_batch_size, crop_size, 0, train_test_rate)

            test_high_img_float32 = test_high_img_float32.astype(np.float32)
            test_low_img_float32 = test_low_img_float32.astype(np.float32)

            ##### forward pass to calculate psnr
            layer3_f = sess.run(layer3_out,
                    feed_dict={in_img_f : test_low_img_float32})
            layer3_uint8 = (layer3_f*255.0).astype(np.uint8)

            ##### calculate psnr
            psnr_ = sess.run(psnr,
                    feed_dict={label_img_uint8 : test_high_img_uint8, out_img_uint8 : layer3_uint8})

            print("itr :"+str(itr)+"  cost :"+str(cost_)+"  psnr :"+str(psnr_))
    #################################### save weights and biases
    saver.save(sess, save_path)
