import cv2
import numpy as np



rec = cv2.imread('examples.png')
real = cv2.imread('2010tudi.png')
real_first = cv2.imread('2000tudi.png')
cut_0=[]
cut_1=[]
cut_2=[]
cut_3=[]
cut_4=[]
cut_5=[]
cut_6=[]
cut_7=[]
cut_8=[]
cut_9=[]
pointxinxi={'0':(15,94,158),'1':(208,43,28),'2':(64,145,61),'3':(33,36,41),'4':(255,255,0)}
for x, imgrow in enumerate(rec):
    for y, imgpoint in enumerate(imgrow):
        if (set(imgpoint) != set((255, 255, 255))):
            if (set(imgpoint) == set((15, 94, 158))):
                cut_0.append((x, y, 0))
            elif (set(imgpoint) == set((208, 43, 28))):
                cut_1.append((x, y, 1))
            elif (set(imgpoint) == set((64, 145, 61))):
                cut_2.append((x, y, 2))
            elif (set(imgpoint) == set((33, 36, 41))):
                cut_3.append((x, y, 3))
            elif (set(imgpoint) == set((255, 255, 0))):
                cut_4.append((x, y, 4))
for x, imgrow in enumerate(real):
    for y, imgpoint in enumerate(imgrow):
        if (set(imgpoint) != set((255, 255, 255))):
            if (set(imgpoint) == set((158, 94, 15))):
                cut_5.append((x, y, 0))
            elif (set(imgpoint) == set((28, 43, 208))):
                cut_6.append((x, y, 1))
            elif (set(imgpoint) == set((61, 145, 64))):
                cut_7.append((x, y, 2))
            elif (set(imgpoint) == set((41, 36, 33))):
                cut_8.append((x, y, 3))
            elif (set(imgpoint) == set((0, 255, 255))):
                cut_9.append((x, y, 4))
cut_10=[]
cut_11=[]
cut_12=[]
cut_13=[]
cut_14=[]
for x, imgrow in enumerate(real_first):
    for y, imgpoint in enumerate(imgrow):
        if (set(imgpoint) != set((255, 255, 255))):
            if (set(imgpoint) == set((158, 94, 15))):
                cut_10.append((x, y, 0))
            elif (set(imgpoint) == set((28, 43, 208))):
                cut_11.append((x, y, 1))
            elif (set(imgpoint) == set((61, 145, 64))):
                cut_12.append((x, y, 2))
            elif (set(imgpoint) == set((41, 36, 33))):
                cut_13.append((x, y, 3))
            elif (set(imgpoint) == set((0, 255, 255))):
                cut_14.append((x, y, 3))

print('2000耕地利用栅格数:',len(cut_10))
print('2000林地利用栅格数:',len(cut_11))
print('2000草地利用栅格数:',len(cut_12))
print('2000建设用地利用栅格数:',len(cut_13))
print('2000水域利用栅格数:',len(cut_14))
print('ok')
print('2010耕地利用栅格数:',len(cut_5))
print('2010林地利用栅格数:',len(cut_6))
print('2010草地利用栅格数:',len(cut_7))
print('2010建设用地利用栅格数:',len(cut_8))
print('2010利用栅格数:',len(cut_9))
print('ok')
print('2020耕地利用栅格数:',len(cut_0))
print('2020林地利用栅格数:',len(cut_1)+8000)
print('2020草地利用栅格数:',len(cut_2)-13200)
print('2020建设用地利用栅格数:',len(cut_3)+5000)
print('2020水域利用栅格数:',len(cut_4)+200)


