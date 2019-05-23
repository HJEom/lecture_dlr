import tensorflow as tf
import numpy as np
import cv2   # (h, w, c) , cv2.imread fcn stores image in (h, w, BGR) by default.
import sys
import os

try:
    os.path.exists(sys.argv[1])
except:
    print("usage : $1 /<dataset>/<directory>/<path>/SR_dataset/Set5/")
    print("usage : $2 /<parameter>/<path>/params/")
    sys.exit(1)

test_img_path = sys.argv[1]
params_path = sys.argv[2]

#################################### to save tested images.
if os.path.isdir(params_path + "/../tested_images") == False:
    os.makedirs(params_path +  "/../tested_images")
tested_image_path = params_path +  "/../tested_images/"

def get_all_dataset(image_path, image_list):
    high_images = np.empty(len(image_list), dtype=object)
    low_images = np.empty(len(image_list), dtype=object)
    print("\ngetting all dataset for train and test...")
    for i in range(len(image_list)):
        image = cv2.imread(image_path+'/'+image_list[i])
        high_images[i] = tf.reshape(tf.image.rgb_to_grayscale(image),[image.shape[0],image.shape[1],1])
        low_images[i] = tf.image.resize_images(tf.image.resize_images(tf.reshape(tf.image.rgb_to_grayscale(image),[image.shape[0],image.shape[1],1]), (int(image.shape[0]/2),int(image.shape[1]/2)), method=tf.image.ResizeMethod.BICUBIC), (image.shape[0],image.shape[1]), method=tf.image.ResizeMethod.BICUBIC)
    return high_images, low_images

def standardization(images):
    img_avr = np.mean(images, axis=(0,1,2), keepdims=True)
    img_var = np.var(images, axis=(0,1,2), keepdims=True)
    images = (images-img_avr)/img_var
    return images, img_avr, img_var

sess = tf.Session()
saver = tf.train.import_meta_graph(params_path+'./train.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint(params_path))

graph = tf.get_default_graph()
#################################### print tensor_name
#for i in graph.get_operations():
#    print(i.name)

#################################### print variables
#w1 = graph.get_tensor_by_name("w1:0")
#print("tensor shape : ", w3.get_shape())
#w1_, w2_, w3_, b1_, b2_, b3_ = sess.run([w1,w2,w3,b1,b2,b3])


#################################### for forward
in_img_f = graph.get_tensor_by_name("in_img_f:0")
label_img_f = graph.get_tensor_by_name("label_img_f:0")

#################################### for psnr
out_img_uint8 = graph.get_tensor_by_name("out_img_uint8:0")
label_img_uint8 = graph.get_tensor_by_name("label_img_uint8:0")

#################################### results
layer3_out = graph.get_tensor_by_name("layer3_out/add:0")

#################################### psnr
psnr = graph.get_tensor_by_name("psnr/Mean:0")

#################################### fetch dataset for test
test_img_list = os.listdir(test_img_path)

with tf.device('/cpu:0'):
    high_img, low_img = get_all_dataset(test_img_path, test_img_list)
    print("convert tensor to numpy array...")
    for number_img in range(len(test_img_list)):
        high_img[number_img] = (sess.run(high_img[number_img])).reshape(1,high_img[number_img].shape[0],high_img[number_img].shape[1],high_img[number_img].shape[2])
        low_img[number_img] = (sess.run(low_img[number_img])).reshape(1,low_img[number_img].shape[0],low_img[number_img].shape[1],low_img[number_img].shape[2])

for test_set_number in range(high_img.shape[0]):
    #################################### forward pass to calculate psnr and to save result img.
    high_img_float32 = (high_img[test_set_number]/255.0).astype(np.float32)    # for original version
    low_img_float32 = (low_img[test_set_number]/255.0).astype(np.float32)    # for original version
#    high_img_float32, _, _ = standardization(high_img[test_set_number])    # for attemps01 version
#    low_img_float32, l_avr, l_var  = standardization(low_img[test_set_number])    # for attemps01 version

    high_img_float32 = high_img_float32.astype(np.float32)
    ow_img_float32 = low_img_float32.astype(np.float32)

    test_out_img_f = sess.run(layer3_out,
            feed_dict={in_img_f : low_img_float32})
    test_out_img_uint8 = (test_out_img_f*255.0).astype(np.uint8)    # for original version
#    test_out_img_uint8 = ((test_out_img_f*l_var)+l_avr).astype(np.uint8)    # for attemps01 version

    #################################### calculate psnr
    psnr_ = sess.run(psnr,
            feed_dict={label_img_uint8 : high_img[test_set_number], out_img_uint8 : test_out_img_uint8})
    print("image name :", test_img_list[test_set_number], "  psnr :", psnr_)
    
    #################################### save result images
    test_out_img_uint8 = test_out_img_uint8.reshape(test_out_img_uint8.shape[1],test_out_img_uint8.shape[2],1)    # reduce the dims from (1, h, w, 1) to (h, w, 1)
    cv2.imwrite(tested_image_path + 'out_img_' + str(test_img_list[test_set_number])[:6] + '.jpg', test_out_img_uint8[:,:,:])

    low_img[test_set_number] = low_img[test_set_number].reshape(low_img[test_set_number].shape[1], low_img[test_set_number].shape[2], 1)
    cv2.imwrite(tested_image_path + 'in_img_' + str(test_img_list[test_set_number])[:6] + '.jpg', low_img[test_set_number][:,:,:])
    
    high_img[test_set_number] = high_img[test_set_number].reshape(high_img[test_set_number].shape[1], high_img[test_set_number].shape[2], 1)
    cv2.imwrite(tested_image_path + 'label_img_' + str(test_img_list[test_set_number])[:6] + '.jpg', high_img[test_set_number][:,:,:])

