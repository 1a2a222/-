import cv2
import numpy as np
railimg=cv2.imread("railway.png")#主意，彩色图像读入后有三个维度，第一个是高，第二个是宽，第三个是通道数
roadimg=cv2.imread("road.png")
quimg=cv2.imread("区中心.png")
shiimg=cv2.imread("市中心.png")
roadimg=cv2.imread("road.png")
img_2000=cv2.imread("2000tudi.png")
img_2010=cv2.imread("2010tudi.png")


cutimgw=21#必须是奇数，截取图片的高宽
cutimgh=21


def cutphoto(img):
    cutimg_all = []
    for x,imgrow in enumerate(img):
        for y,imgpoint in enumerate(imgrow):
            if(set(imgpoint)!=set([255,255,255])):
                cutimg = np.zeros([21,21,3])
                cutimg = img[x-11:x+10,y-11:y+10,:]
                cutimg_all.append(cutimg)
#                print(cutimg_all)
#                cv2.imshow("cutimg",cutimg)
#                cv2.waitKey (0)
#                cv2.destroyAllWindows() 
    return cutimg_all
            
#从此点处扩展为一副小图片

#railimg1,roadimg2,quimg3,shiimg4,img2000_5,img2010_6 = np.zeros((cutimgh,cutimgw,3))
#railimg1 = myimg[x-int(cutimgh/2):x+int(cutimgh/2),y-int(cutimgw/2):y+int(cutimgw/2),:]
#        roadimg2 = myimg[x - int(cutimgh / 2):x + int(cutimgh / 2), y - int(cutimgw / 2):y + int(cutimgw / 2), :]
#        quimg3 = myimg[x - int(cutimgh / 2):x + int(cutimgh / 2), y - int(cutimgw / 2):y + int(cutimgw / 2), :]
#        shiimg4 = myimg[x - int(cutimgh / 2):x + int(cutimgh / 2), y - int(cutimgw / 2):y + int(cutimgw / 2), :]
#        img2000_5 = myimg[x - int(cutimgh / 2):x + int(cutimgh / 2), y - int(cutimgw / 2):y + int(cutimgw / 2), :]
#        img2010_6 = myimg[x - int(cutimgh / 2):x + int(cutimgh / 2), y - int(cutimgw / 2):y + int(cutimgw / 2), :]
        
        
#             a=tf[21 21 3 5]
#                a[:,:,:1]=newimg1
#                a[2]=newimg2
#                a..[5]=newimg5
#                tf.reshape[21,21,15]


  
cut = cutphoto(railimg)



#cutphoto(roadimg)
#cutphoto(quimg)
#cutphoto(shiimg)
#cutphoto(roadimg)
#cutphoto(img_2000)
#cutphoto(img_2010)
