# -*- coding: utf-8 -*-
"""
Created on Mon May  7 19:09:52 2018
@author: fyl
"""

#卷积神经网络
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import os
railimg=cv2.imread("railway.png")#主意，彩色图像读入后有三个维度，第一个是高，第二个是宽，第三个是通道数
roadimg=cv2.imread("road.png")
quimg=cv2.imread("qucenter.png")
shiimg=cv2.imread("shicenter.png")

img_2000=cv2.imread("2000tudi.png")
img_2010=cv2.imread("2010tudi.png")

cutimgw=21#必须是奇数，截取图片的高宽
cutimgh=21


###裁剪图像，裁剪出21*21*3的训练图像
def cutphoto(img):
    cutimg_all = []
    for x,imgrow in enumerate(img):
        for y,imgpoint in enumerate(imgrow):
            if(set(imgpoint)!=set((255,255,255))):
                cutimg = img[x-10:x+11,y-10:y+11,:]
                cutimg_all.append(cutimg)
                #print(cutimg.shape)
#                print(cutimg_all)
#                cv2.imshow("cutimg",cutimg)
#                cv2.waitKey (0)
#                cv2.destroyAllWindows()
    return cutimg_all

###计算图像的准确率
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

###需要是nparray图片数据,把数据归一化到-1到1
def img_normalized(imagedata):
    return np.array(imagedata) / 127.5 - 1.#返回的是float64

###防止梯度爆炸或梯度消失
class batch_norm(object):#batchnorm用于在每一层输入的时候，再加个预处理（1.标准正太归一化2.向原正太方差，平均值做拉伸）操作，作用是减少“梯度弥散”，何为梯度弥散？就是在层数过多的情况下信息过大或者过小的情况(如0.1的100次方太小了,1.1的100次方太大),也称梯度消失和爆炸
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon#用于防止除以零的极小值
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):#只要定义类型的时候，实现__call__函数，这个类型就成为可调用的。换句话说，我们可以把这个类型的对象当作函数来使用，相当于 重载了括号运算符。
    return tf.contrib.layers.batch_norm(x,#使用方法 定义：self.g_bn3 = batch_norm(name='g_bn3')  使用：h3 = tf.nn.relu(self.g_bn3(h3, train=False)) h3是某层输出结果，让其通过g_bn3把结果平滑化
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

###卷积层的定义，采用5*5的卷积核
def conv2dwj(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
      w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
      conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
      biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
      conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
      return conv

def findmax(array):
    max=-999
    maxi=0
    for i in range(imgno):
        if array[i]>max:
            max=array[i]
            maxi=i
    return maxi

#图像裁剪，训练因子列表
railimg_cut_all = cutphoto(railimg)
roadimg_cut_all = cutphoto(roadimg)
quimg_cut_all = cutphoto(quimg)
shiimg_cut_all = cutphoto(shiimg)
img_2000_cut_all = cutphoto(img_2000)
print((img_2000_cut_all[0][0][0]))
print('railimg_cut_all',len(railimg_cut_all))
print('railimg_cut_all[0]',len(railimg_cut_all[0]))
print('railimg_cut_all[0][0]',len(railimg_cut_all[0][0]))
print('railimg_cut_all[0][0][0]',len(railimg_cut_all[0][0][0]))
###
print('roadimg_cut_all',len(roadimg_cut_all))
print('roadimg_cut_all[0]',len(roadimg_cut_all[0]))
print('roadimg_cut_all[0][0]',len(roadimg_cut_all[0][0]))
print('roadimg_cut_all[0][0][0]',len(roadimg_cut_all[0][0][0]))
###
print('quimg_cut_all',len(quimg_cut_all))
print('quimg_cut_all[0]',len(quimg_cut_all[0]))
print('quimg_cut_all[0][0]',len(quimg_cut_all[0][0]))
print('quimg_cut_all[0][0][0]',len(quimg_cut_all[0][0][0]))
###
print('shiimg_cut_all',len(shiimg_cut_all))
print('shiimg_cut_all[0]',len(shiimg_cut_all[0]))
print('shiimg_cut_all[0][0]',len(shiimg_cut_all[0][0]))
print('shiimg_cut_all[0][0][0]',len(shiimg_cut_all[0][0][0]))
###
print('img_2000_cut_all',len(img_2000_cut_all))
print('img_2000_cut_all[0]',len(img_2000_cut_all[0]))
print('img_2000_cut_all[0][0]',len(img_2000_cut_all[0][0]))
print('img_2000_cut_all[0][0][0]',len(img_2000_cut_all[0][0][0]))
###

# print('第一张21*21的图像',len(railimg_cut_all[3600]))
# print('hello1')
# print('第一个像素点的rgb',railimg_cut_all[3600][0])
# print('hello2')
# print(railimg_cut_all[3600][0][0])##去切片的21*21*3的第一个像素的rgb值
# print('hell3o')
# print(railimg_cut_all[3600][0][0][0])
# print ('helllo4')
# print(railimg_cut_all[3600][0][0][1])
# print('hellos5')
# print(railimg_cut_all[3600][0][0][2])
# print('helld6')
###训练图

###末期年有色彩点的信息，位置信息和相应像素对应的土地类型
img_2010_cut_all = []
print("bbb")

# pointxinxi={'1':[158,94,15],'2':[28,43,208],'3':[61,145,64],'4':[41,36,33],'5':[0,255,255]}
pointxinxi={'0':(158,94,15),'1':(28,43,208),'2':(61,145,64),'3':(41,36,33),'4':(0,255,255)}
###采集2010土地信息
for x, imgrow in enumerate(img_2010):
    for y, imgpoint in enumerate(imgrow):
        if (set(imgpoint) != set((255, 255, 255))):
                # img_2010_cut_all.append((x,y,1))
            # if (set(imgpoint) == set([158, 94, 15])):
            #     img_2010_cut_all.append([x, y, 1])
            # elif (set(imgpoint) == set([28, 43, 208])):
            #     img_2010_cut_all.append([x, y, 2])
            # elif (set(imgpoint) == set([61, 145, 64])):
            #     img_2010_cut_all.append([x, y, 3])
            # elif (set(imgpoint) == set([41, 36, 33])):
            #     img_2010_cut_all.append([x, y, 4])
            # elif (set(imgpoint) == set([0, 255, 255])):
            #     img_2010_cut_all.append([x, y, 5])
            if (set(imgpoint) == set((158, 94, 15))):
                img_2010_cut_all.append((x, y, 0))
            elif (set(imgpoint) == set((28, 43, 208))):
                img_2010_cut_all.append((x, y, 1))
            elif (set(imgpoint) == set((61, 145, 64))):
                img_2010_cut_all.append((x, y, 2))
            elif (set(imgpoint) == set((41, 36, 33))):
                img_2010_cut_all.append((x, y, 3))
            elif (set(imgpoint) == set((0, 255, 255))):
                img_2010_cut_all.append((x, y, 4))


print('img_2010_cut_all',len(img_2000_cut_all))
print('img_2010_cut_all[0]',len(img_2000_cut_all[0]))
print('img_2010_cut_all[0][0]',len(img_2000_cut_all[0][0]))
print('img_2010_cut_all[0][0][0]',len(img_2000_cut_all[0][0][0]))



#定义placeholder
sess = tf.Session()
imgno=5
xs = tf.placeholder(tf.float32,[1,cutimgh,cutimgw,3,imgno],name='x_input')#21*21的图像
ys = tf.placeholder(tf.int64,[1,],name='y_input')


x_image = tf.reshape(xs,[1,cutimgh,cutimgw,-1])#channel:1
print('---------------------------')
print(x_image.shape)###应该是21*21*3*5或者是21*21*3*4


###防止卷积层的梯度爆炸
convlayer_bn1 = batch_norm(name='convlayer_bn1')
convlayer_bn2 = batch_norm(name='convlayer_bn2')
convlayer_bn3 = batch_norm(name='convlayer_bn3')
###conv1layer，存储命名空间
with tf.variable_scope("convlayer") as scope:
    h0 = tf.nn.leaky_relu(convlayer_bn1(conv2dwj(x_image, 32,name='c_h0')))
    print(h0.shape)
    h1 = tf.nn.leaky_relu(convlayer_bn2(conv2dwj(h0, 64,name='c_h1')))
    print(h1.shape)
    d0=tf.reshape(h1, [1,-1])
    print(d0.shape)
    d1=tf.layers.dense(d0,512,tf.nn.leaky_relu,name='d_h1')
    print(d1.shape)
    d2=tf.layers.dense(d1,256,tf.nn.leaky_relu,name='d_h2')
    print(d2.shape)
    d3=tf.layers.dense(d2,imgno,name='d_h3')
    print('d3',d3.shape)
    print('ys',ys.shape)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(d3),reduction_indices=[1]))

###计算交叉熵来获取损失值

cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=d3, labels=ys))
###通过训练函数降低损失值

train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
###初始化所有变量
init = tf.initialize_all_variables()
sess.run(init)
print("aaa")
writer = tf.summary.FileWriter("logs/",sess.graph)
####将数据导入imagedata里面存储
count1=1
count2=1
count3=1
count4=1
count5=1
count6=1

###定义xdata数据
imagedata = np.zeros((90122,21, 21, 3, 5))
##存储railimg_cut_all数据
for i in range(90122):#90156
    # imagedata=[]
    # imagedata.append(railimg_cut_all[i])
    # imagedata.append(roadimg_cut_all[i])
    # imagedata.append(quimg_cut_all[i])
    # imagedata.append(shiimg_cut_all[i])
    # imagedata.append(img_2000_cut_all[i])
    for j in range(21):
        for k in range(21):
            for l in range(3):
                    imagedata[i][j][k][l][0] = railimg_cut_all[i][j][k][l]
                    # count1 += 1
                    # print(count1)
print('完成railimg_cut_all数据的存储')

###存储roadimg_cut_all数据
for i in range(90122):#90156
    # imagedata=[]
    # imagedata.append(railimg_cut_all[i])
    # imagedata.append(roadimg_cut_all[i])
    # imagedata.append(quimg_cut_all[i])
    # imagedata.append(shiimg_cut_all[i])
    # imagedata.append(img_2000_cut_all[i])
    for j in range(21):
        for k in range(21):
            for l in range(3):
                imagedata[i][j][k][l][1] = roadimg_cut_all[i][j][k][l]
                # count2 += 1
                # print(count2)
print('完成roadimg_cut_all数据的存储')

###存储quimg_cut_all数据
for i in range(90122):#90159
    # imagedata=[]
    # imagedata.append(railimg_cut_all[i])
    # imagedata.append(roadimg_cut_all[i])
    # imagedata.append(quimg_cut_all[i])
    # imagedata.append(shiimg_cut_all[i])
    # imagedata.append(img_2000_cut_all[i])
    for j in range(21):
        for k in range(21):
            for l in range(3):
                imagedata[i][j][k][l][2] = quimg_cut_all[i][j][k][l]
                # count3 += 1
                # print(count3)
print('完成quimg_cut_all数据的存储')

###存储shiimg_cut_all数据
for i in range(90122):#90133
    # imagedata=[]
    # imagedata.append(railimg_cut_all[i])
    # imagedata.append(roadimg_cut_all[i])
    # imagedata.append(quimg_cut_all[i])
    # imagedata.append(shiimg_cut_all[i])
    # imagedata.append(img_2000_cut_all[i])
    for j in range(21):
        for k in range(21):
            for l in range(3):
                imagedata[i][j][k][l][3] = shiimg_cut_all[i][j][k][l]
                # count4 += 1
                # print(count4)
print('完成shiimg_cut_all数据的存储')

###存储img_2000_cut_all数据
for i in range(90122):#90122
    # imagedata=[]
    # imagedata.append(railimg_cut_all[i])
    # imagedata.append(roadimg_cut_all[i])
    # imagedata.append(quimg_cut_all[i])
    # imagedata.append(shiimg_cut_all[i])
    # imagedata.append(img_2000_cut_all[i])
    for j in range(21):
        for k in range(21):
            for l in range(3):
                imagedata[i][j][k][l][4] = img_2000_cut_all[i][j][k][l]
                # count5 += 1
                # print(count5)
print('完成img_2000_cut_all数据的存储')
#
#
pointlabel = []
# ###存储ydata数据
for i in range(90122):#90122
    ###添加维度
    pointlabel.append(img_2010_cut_all[i][2])
    #print(img_2010_cut_all[i][2])
print('完成img_2010_cut_all数据的存储')

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
# -----------------------------------------------TRIAN
##计算交叉熵来获取损失值
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=d3, labels=ys))
###通过训练函数降低损失值
train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
if os.path.isfile('./my_test_model.meta'):
    saver.restore(sess,'./my_test_model')
epoch=100
for j in range(10):
    print(j)
    for i in range(100):
        sr_imagedata = [img_normalized(imagedata[i])]
        #print(sr_imagedata)
        sr_polintlabel = [pointlabel[i]]
        #print(sr_imagedata[0].shape)
    #print(imagedata)
    ####给2010用地信息添加维度
    #print(pointlabel)
        sess.run([train_step],feed_dict={xs:sr_imagedata,ys:sr_polintlabel})
        err_cnn = cross_entropy.eval(session=sess,
                                 feed_dict={xs: sr_imagedata, ys:sr_polintlabel})

    print(err_cnn)
    saver.save(sess, './my_test_model')

# -----------------------------------------------USE
saver.restore(sess,'./my_test_model')
newimg=np.zeros((600,600,3),dtype='uint8' )
errorno=0
for i in range(90122):
    sr_imagedata = [img_normalized(imagedata[i])]
    rec_result=sess.run([d3],feed_dict={xs:sr_imagedata})
    #print(rec_result)
    rec_label=str(findmax(rec_result[0][0]))
    #print(rec_label,img_2010_cut_all[i][2],pointxinxi[rec_label])
    if(rec_label!=str(img_2010_cut_all[i][2])):
        errorno=errorno+1
    x=img_2010_cut_all[i][0]
    y=img_2010_cut_all[i][1]
    newimg[x][y]=np.uint8(pointxinxi[rec_label])
    #print('110')
cv2.imshow("2010rec",newimg)
cv2.waitKey(0)
print('ok')
cv2.imwrite('examples3.png', newimg,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
print('ookk')
# cv2.imwrite("./yuc.jpg", newimg,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
