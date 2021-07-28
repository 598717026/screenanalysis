# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:12:14 2021

@author: 20206161
"""

import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
#from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis
import numpy as np
import os
import csv
import time


'''
def analysis(x, y, w, h, gray, r, g, b, sf = "", vh = 'h'):

    (imgh, imgw) = gray.shape[:2]
    
    print("analysis", x, y, w, h, sf)
    posw = w
    posh = h
    
    if x == 0:
        x = 1
    if y == 0:
        y = 1
    
    if posw == 0:
        posw = 1
    if posw > x:
        posw = x
    if posw + x >= imgw:
        posw = imgw - x
    if posh == 0:
        posh = 1
    if posh > y:
        posh = y
    if posh + y >= imgh:
        posh = imgh - y
        
    
    imcutgray = gray[y - posh : y + posh - 1,x - posw : x + posw - 1]
    np.savetxt(os.path.splitext(sf)[0] + ".gray.csv", imcutgray, delimiter=',')
    
    imcutr = r[y - (posh - 1) : y + posh,x - (posw - 1) : x + posw]
    np.savetxt(os.path.splitext(sf)[0] + ".r.csv", imcutr, delimiter=',')
    imcutg = g[y - (posh - 1) : y + posh,x - (posw - 1) : x + posw]
    np.savetxt(os.path.splitext(sf)[0] + ".g.csv", imcutg, delimiter=',')
    imcutb = b[y - (posh - 1) : y + posh,x - (posw - 1) : x + posw]
    np.savetxt(os.path.splitext(sf)[0] + ".b.csv", imcutb, delimiter=',')
    
    if posh == 1:
        
#                fig, ax = plt.subplots(2, 2)
    
        da = imcutgray[:, : -2].astype(np.int32)
        db = imcutgray[:, 1 : -1].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        
        grayFft = df
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".gray.delta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".gray.fft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
#                ax[0][0].plot(np.arange(0,np.size(df),1),(df - np.min(df)) / (np.max(df) - np.min(df)).flatten())
#                ax[0][0].set_ylabel('gray')
    
        da = imcutr[:, : -2].astype(np.int32)
        db = imcutr[:, 1 : -1].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        redFft = df
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".r.delta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".r.fft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
    
#                ax[0][1].plot(np.arange(0,np.size(df),1),(df - np.min(df)) / (np.max(df) - np.min(df)).flatten(), color='r')
#                ax[0][1].set_ylabel('red')
    
        da = imcutg[:, : -2].astype(np.int32)
        db = imcutg[:, 1 : -1].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        greenFft = df
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".g.delta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".g.fft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
    
#                ax[1][0].plot(np.arange(0,np.size(df),1),(df - np.min(df)) / (np.max(df) - np.min(df)).flatten(), color='g')
#                ax[1][0].set_ylabel('green')
    
        da = imcutb[:, : -2].astype(np.int32)
        db = imcutb[:, 1 : -1].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        blueFft = df
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".b.delta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".b.fft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
#                ax[1][1].plot(np.arange(0,np.size(df),1),(df - np.min(df)) / (np.max(df) - np.min(df)).flatten(), color='b')
#                ax[1][1].set_ylabel('blue')
    
        
        #plt.show()
    
#                plt.savefig(os.path.splitext(sf)[0] + ".fft.jpg")
    
    elif posw == 1:
        
#                fig, ax = plt.subplots(2, 2)
    
        da = imcutgray[ : -2, :].astype(np.int32)
        db = imcutgray[ 1 : -1, :].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        grayFft = df
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".gray.delta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".gray.fft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
#                ax[0][0].plot(np.arange(0,np.size(df),1),(df - np.min(df)) / (np.max(df) - np.min(df)).flatten())
#                ax[0][0].set_ylabel('gray')
    
        da = imcutr[ : -2, :].astype(np.int32)
        db = imcutr[ 1 : -1, :].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        redFft = df
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".r.delta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".r.fft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
#                ax[0][1].plot(np.arange(0,np.size(df),1),(df - np.min(df)) / (np.max(df) - np.min(df)).flatten(), color='r')
#                ax[0][1].set_ylabel('red')
    
        da = imcutg[ : -2, :].astype(np.int32)
        db = imcutg[ 1 : -1, :].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        greenFft = df
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".g.delta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".g.fft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
#                ax[1][0].plot(np.arange(0,np.size(df),1),(df - np.min(df)) / (np.max(df) - np.min(df)).flatten(), color='g')
#                ax[1][0].set_ylabel('green')
    
        da = imcutb[ : -2, :].astype(np.int32)
        db = imcutb[ 1 : -1, :].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        blueFft = df
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".b.delta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".b.fft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
#                ax[1][1].plot(np.arange(0,np.size(df),1),(df - np.min(df)) / (np.max(df) - np.min(df)).flatten(), color='b')
#                ax[1][1].set_ylabel('blue')
    
        #plt.show()
    
#                plt.savefig(os.path.splitext(sf)[0] + ".fft.jpg")
    
    else:
        da = imcutgray[ : -2, : -2].astype(np.int32)
        db = imcutgray[ 1 : -1, 1 : -1].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        grayFft = df
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".gray.xydelta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".gray.xyfft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
        da = imcutr[ : -2, : -2].astype(np.int32)
        db = imcutr[ 1 : -1, 1 : -1].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        redFft = df
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".r.xydelta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".r.xyfft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
        da = imcutg[ : -2, : -2].astype(np.int32)
        db = imcutg[ 1 : -1, 1 : -1].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        greenFft = df
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".g.xydelta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".g.xyfft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
        da = imcutb[ : -2, : -2].astype(np.int32)
        db = imcutb[ 1 : -1, 1 : -1].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        blueFft = df
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".b.xydelta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".b.xyfft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
        ############################################
        
        da = imcutgray[ : , : -2].astype(np.int32)
        db = imcutgray[ :, 1 : -1].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".gray.xdelta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".gray.xfft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
        da = imcutr[ : , : -2].astype(np.int32)
        db = imcutr[ : , 1 : -1].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".r.xdelta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".r.xfft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
        da = imcutg[ : , : -2].astype(np.int32)
        db = imcutg[ : , 1 : -1].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".g.xdelta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".g.xfft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
        da = imcutb[ : , : -2].astype(np.int32)
        db = imcutb[ : , 1 : -1].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".b.xdelta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".b.xfft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
        ####################################################
        
        da = imcutgray[ : -2, :].astype(np.int32)
        db = imcutgray[ 1 : -1, :].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".gray.ydelta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".gray.yfft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
        da = imcutr[ : -2, : ].astype(np.int32)
        db = imcutr[ 1 : -1, : ].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".r.ydelta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".r.yfft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
        da = imcutg[ : -2, : ].astype(np.int32)
        db = imcutg[ 1 : -1, :].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".g.ydelta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".g.yfft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
        da = imcutb[ : -2, : ].astype(np.int32)
        db = imcutb[ 1 : -1, : ].astype(np.int32)
        dd = db - da
        df = abs(np.fft.rfft(dd))
        #print(da, db, dd, df)
        np.savetxt(os.path.splitext(sf)[0] + ".b.ydelta.csv", dd, delimiter=',')
        np.savetxt(os.path.splitext(sf)[0] + ".b.yfft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
        
    return grayFft, redFft, greenFft, blueFft
'''

# 闪点分析
def FlashAnalysis(x, y, w, h, gray, sf):
    
    (imgh, imgw) = gray.shape[:2]
    
    posw = w
    posh = h
    
    if x == 0:
        x = 1
    if y == 0:
        y = 1
    
    if posw == 0:
        posw = 1
    if posw > x:
        posw = x
    if posw + x >= imgw:
        posw = imgw - x
    if posh == 0:
        posh = 1
    if posh > y:
        posh = y
    if posh + y >= imgh:
        posh = imgh - y
        
    print(time.time())
    imcutgray = gray[y - (posh - 1) : y + posh,x - (posw - 1) : x + posw].flatten()
    
    print(time.time())
    gavg = np.mean(imcutgray)
    gmax = np.max(imcutgray)
    gmin = np.min(imcutgray)
    print(time.time())
    gstd = np.std(imcutgray, ddof = 1)
    print(time.time())
    hist ,bins = np.histogram(imcutgray,bins=256,range=(0,256))
    histogram = []
    for i in range(len(hist)):
        histogram.append([i, hist[i]])
        
    print(time.time())
    #print(histogram)
        
    with open(sf, "w", newline='') as csvfile: 
        writer = csv.writer(csvfile)
     
        #先写入columns_name
        #writer.writerow(["平均值","最大值","最小值", "标准差"])
        #print([str(gavg), str(gmax), str(gmin), str(gstd)])
        #writer.writerows(["+ %f"%(gavg), "+ %f"%(gmax), "+ %f"%(gmin), "+ %f"%(gstd)])
        
        writer.writerow(["平均值", gavg])
        writer.writerow(["最大值", gmax])
        writer.writerow(["最小值", gmin])
        writer.writerow(["标准差", gstd])
        #写入多行用writerows
        writer.writerow(["直方图"])
        writer.writerow(["亮度值","统计"])
        writer.writerows(histogram)
    
    return gavg, gmax, gmin, gstd, histogram
    
# MTF分析
def analysis(x, y, w, h, gray, r, g, b, sf = "", vh = 'h'):

    (imgh, imgw) = gray.shape[:2]
    
    print("analysis", x, y, w, h, sf)
    posw = w
    posh = h
    
    if x == 0:
        x = 1
    if y == 0:
        y = 1
    
    if posw == 0:
        posw = 1
    if posw > x:
        posw = x
    if posw + x >= imgw:
        posw = imgw - x
    if posh == 0:
        posh = 1
    if posh > y:
        posh = y
    if posh + y >= imgh:
        posh = imgh - y
        
    imcutgray = gray[y - (posh - 1) : y + posh,x - (posw - 1) : x + posw]
    #np.savetxt(os.path.splitext(sf)[0] + ".gray.csv", imcutgray, delimiter=',')
    
    imcutr = r[y - (posh - 1) : y + posh,x - (posw - 1) : x + posw]
    #np.savetxt(os.path.splitext(sf)[0] + ".r.csv", imcutr, delimiter=',')
    imcutg = g[y - (posh - 1) : y + posh,x - (posw - 1) : x + posw]
    #np.savetxt(os.path.splitext(sf)[0] + ".g.csv", imcutg, delimiter=',')
    imcutb = b[y - (posh - 1) : y + posh,x - (posw - 1) : x + posw]
    #np.savetxt(os.path.splitext(sf)[0] + ".b.csv", imcutb, delimiter=',')
    
    #横向分析
    if vh == 'h':
        dc = np.ones(imcutgray.shape[1], dtype=np.int32)
        dc -= dc
        for i in range(imcutgray.shape[0]):
            dc = dc + imcutgray[i, :]
        dc = dc / imcutgray.shape[0]
            
    else:
        dc = np.ones(imcutgray.shape[0], dtype=np.int32)
        dc -= dc
        for i in range(imcutgray.shape[1]):
            dc = dc + imcutgray[ : , i]
        dc = dc / imcutgray.shape[1]
            
    da = dc[ : -2].astype(np.int32)
    db = dc[1 : -1].astype(np.int32)
    dd = db - da
    df = abs(np.fft.rfft(dd))
    grayFft = df
    
    #np.savetxt(os.path.splitext(sf)[0] + ".gray.delta.csv", dd, delimiter=',')
    np.savetxt(os.path.splitext(sf)[0] + ".gray.fft.csv", (df - np.min(df)) / (np.max(df) - np.min(df)), delimiter=',')
    
    redFft = None
    greenFft = None
    blueFft = None
            
    return grayFft, redFft, greenFft, blueFft
    


class analysisWin(QWidget):
    def __init__(self, parent=None): 
        super(analysisWin, self).__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
    
                
'''自定义的QLabel类'''
class myImgLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super(myImgLabel, self).__init__(parent)
        self.hitX = -10
        self.hitY = -10
        self.hitW = 1
        self.hitH = 1
    
    '''重载一下鼠标按下事件(单击)'''
    def mousePressEvent(self, event):
        return
        if event.buttons () == QtCore.Qt.LeftButton:                           # 左键按下
            #self.setText ("单击鼠标左键的事件: 自己定义")
            print("单击鼠标左键")  # 响应测试语句
        elif event.buttons () == QtCore.Qt.RightButton:                        # 右键按下
            self.setText ("单击鼠标右键的事件: 自己定义")
            print("单击鼠标右键")  # 响应测试语句
        elif event.buttons () == QtCore.Qt.MidButton:                          # 中键按下
            self.setText ("单击鼠标中键的事件: 自己定义")
            print("单击鼠标中键")  # 响应测试语句
        elif event.buttons () == QtCore.Qt.LeftButton | QtCore.Qt.RightButton: # 左右键同时按下
            self.setText ("同时单击鼠标左右键的事件: 自己定义")
            print("单击鼠标左右键")  # 响应测试语句
        elif event.buttons () == QtCore.Qt.LeftButton | QtCore.Qt.MidButton:   # 左中键同时按下
            self.setText ("同时单击鼠标左中键的事件: 自己定义")
            print("单击鼠标左中键")  # 响应测试语句
        elif event.buttons () == QtCore.Qt.MidButton | QtCore.Qt.RightButton:  # 右中键同时按下
            self.setText ("同时单击鼠标右中键的事件: 自己定义")
            print("单击鼠标右中键")  # 响应测试语句
        elif event.buttons () == QtCore.Qt.LeftButton | QtCore.Qt.MidButton \
             | QtCore.Qt.RightButton:                                          # 左中右键同时按下
            self.setText ("同时单击鼠标左中右键的事件: 自己定义")
            print("单击鼠标左中右键")  # 响应测试语句
 
    
    '''重载一下鼠标双击事件'''
    def mouseDoubieCiickEvent(self, event):
        if event.buttons () == QtCore.Qt.LeftButton:                           # 左键按下
            self.setText ("双击鼠标左键的功能: 自己定义")
        self.setText ("鼠标双击事件: 自己定义(%d, %d)"%(event.x(), event.y()))
 
    '''重载一下鼠标双击事件'''
    def mouseCiickEvent(self, event):
        if event.buttons () == QtCore.Qt.LeftButton:                           # 左键按下
            self.setText ("单击鼠标左键的功能: 自己定义")
        self.setText ("鼠标单击事件: 自己定义(%d, %d)"%(event.x(), event.y()))
 
    '''重载一下鼠标键释放事件'''
    def mouseReleaseEvent(self, event):
        #self.setText("鼠标释放事件: 自己定义")
        print("鼠标释放: 自己定义(%d, %d)"%(event.x(), event.y()))
        self.hitX = event.x()
        self.hitY = event.y()
        self.update()
        
 
    '''重载一下鼠标移动事件'''
    def mouseMoveEvent(self, event):
        #self.setText("鼠标移动事件: 自己定义")
        print("鼠标移动")  # 响应测试语句
        
    def resetHitPos(self):
        self.hitX, self.hitY = -100, -100

    def getHitPos(self):
        return self.hitX, self.hitY

    def setHitPos(self, x, y):
        self.hitX, self.hitY = x, y

    def setHitSize(self, w, h):
        self.hitW, self.hitH = w, h
        self.update()
 
    def paintEvent(self, event):
        super().paintEvent(event)
        painter=QPainter(self)
        painter.begin(self)
        painter.setPen(QPen(QColor(255,0,255), 10))
        painter.drawPoint(self.hitX, self.hitY)
        painter.setPen(QPen(QColor(0,255,0), 1))
        rect =QRect(self.hitX - self.hitW, self.hitY - self.hitH, self.hitW * 2, self.hitH * 2)
        #print(rect)
        painter.drawRect(rect)
        painter.end()

class mainWin(QWidget):
    def __init__(self):
        super(mainWin, self).__init__()

        self.label = None
        
        self.resize(600, 400)
        self.setWindowTitle("label显示图片")

        self.img = None
        self.gray = None
        self.red = None
        self.green = None
        self.blue = None
        
        self.file_path = None

        self.widgetW = QWidget()
        self.layoutW = QHBoxLayout()
        self.labelW = QLabel(self)
        self.labelW.setText("width:" + str(1))
        self.spliderW=QSlider(Qt.Horizontal)
        self.spliderW.valueChanged.connect(self.valChangeW)
        self.spliderW.setMinimum(1)#最小值
        self.spliderW.setMaximum(60)#最大值
        self.spliderW.setSingleStep(1)#步长
        self.spliderW.setTickPosition(QSlider.TicksBelow)#设置刻度位置，在下方
        self.spliderW.setTickInterval(5)#设置刻度间隔
        self.layoutW.addWidget(self.labelW)
        self.layoutW.addWidget(self.spliderW)
        self.widgetW.setLayout(self.layoutW)

        self.widgetH = QWidget()
        self.layoutH = QHBoxLayout()
        self.labelH = QLabel(self)
        self.labelH.setText("hight:" + str(1))
        self.spliderH=QSlider(Qt.Horizontal)
        self.spliderH.valueChanged.connect(self.valChangeH)
        self.spliderH.setMinimum(1)#最小值
        self.spliderH.setMaximum(60)#最大值
        self.spliderH.setSingleStep(1)#步长
        self.spliderH.setTickPosition(QSlider.TicksBelow)#设置刻度位置，在下方
        self.spliderH.setTickInterval(5)#设置刻度间隔
        self.layoutH.addWidget(self.labelH)
        self.layoutH.addWidget(self.spliderH)
        self.widgetH.setLayout(self.layoutH)
        
        
        
        self.widgetSel = QWidget()
        self.layoutSel = QHBoxLayout()
        self.btnW = QRadioButton("横向分析")          #实例化一个选择的按钮
        self.btnW.setChecked(True)                     #设置按钮点点击状态
        self.layoutSel.addWidget(self.btnW)     #布局添加组件
        self.btnH = QRadioButton("竖向分析")           #实例化第二个按钮
        self.layoutSel.addWidget(self.btnH)     #布局添加第二个按钮
        self.widgetSel.setLayout(self.layoutSel)   #界面添加 layout

        self.btnGray = QRadioButton("Gray闪点分析") 
        self.layoutSel.addWidget(self.btnGray)     #布局添加第二个按钮
        self.widgetSel.setLayout(self.layoutSel)   #界面添加 layout

        self.btnRed = QRadioButton("Red闪点分析")  
        self.layoutSel.addWidget(self.btnRed)     #布局添加第二个按钮
        self.widgetSel.setLayout(self.layoutSel)   #界面添加 layout

        self.btnGreen = QRadioButton("Green闪点分析") 
        self.layoutSel.addWidget(self.btnGreen)     #布局添加第二个按钮
        self.widgetSel.setLayout(self.layoutSel)   #界面添加 layout

        self.btnBlue = QRadioButton("Blue闪点分析") 
        self.layoutSel.addWidget(self.btnBlue)     #布局添加第二个按钮
        self.widgetSel.setLayout(self.layoutSel)   #界面添加 layout

        self.label = myImgLabel(self)
        self.label.setText("   显示图片")
        #self.label.setFixedSize(300, 200)
        #self.label.move(160, 160)

        btn = QPushButton(self)
        btn.setText("打开图片")
        btn.clicked.connect(self.openimage)
        
        analysisBtn = QPushButton(self)
        analysisBtn.setText("分析")
        analysisBtn.clicked.connect(self.analysis)
        
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.label)
        layout = QVBoxLayout()
        layout.addWidget(btn)
        layout.addWidget(analysisBtn)
        layout.addWidget(self.widgetW)
        layout.addWidget(self.widgetH)
        layout.addWidget(self.widgetSel)
        layout.addWidget(self.scroll)
        self.setLayout(layout)
        
    def HitSizeChange(self):
        pass
        #print(self.label)
        if self.label is None:
            return
            
        self.label.setHitSize(self.spliderW.value(), self.spliderH.value())
        
        
    def valChangeW(self):
        #print(self.spliderW.value())
        self.labelW.setText("width:" + str(self.spliderW.value()))
        self.HitSizeChange()

    def valChangeH(self):
        #print(self.spliderH.value())
        self.labelH.setText("hight:" + str(self.spliderH.value()))
        self.HitSizeChange()

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;*.png;*.bmp;;All Files(*)")
        if imgName == '':
            return
            
        img = QImage(imgName)
        if img.isNull():
            QMessageBox.information(self, '图像浏览器', '不能加载文件%s.' % filename)
            return
        
        self.file_path = imgName
        
        ptr = img.constBits()
        ptr.setsize(img.byteCount())
        self.img = np.array(ptr).reshape( img.height(), img.width(), 4)
        self.gray = 0.299*self.img[:,:,0]+0.587*self.img[:,:,1]+0.114*self.img[:,:,2]
        self.red = self.img[:,:,0]
        self.green = self.img[:,:,1]
        self.blue = self.img[:,:,2]
        
        jpg = QtGui.QPixmap(img)
        print(jpg.width(), jpg.height())
        self.label.setMinimumSize(jpg.width(), jpg.height())
        self.label.setFixedSize(jpg.width(), jpg.height())
        self.label.setPixmap(jpg)
        self.label.resetHitPos()
        self.spliderW.setMaximum(jpg.width() // 2)#最大值
        self.spliderH.setMaximum(jpg.height() // 2)#最大值

    def analysis(self):
        if self.file_path is None:
            return
    
        x, y = self.label.getHitPos()
        w = self.spliderW.value()
        h = self.spliderH.value()
        
        fileName_choose, filetype = QFileDialog.getSaveFileName(self,  
                            "文件保存",  
                            os.path.basename(self.file_path) + "." + str(x) + "-" + str(y) + "-" + str(w) + "-" + str(h), # 起始路径 
                            "Csv Files (*.csv)")  
        if fileName_choose == "":
            print("\n取消选择")
            return
        sf = fileName_choose
    
        if self.btnW.isChecked() == True:
            vh = 'h'
            analysis(x, y, w, h, self.gray, self.red, self.green, self.blue, sf, vh)
        elif self.btnH.isChecked() == True:
            vh = 'v'
            analysis(x, y, w, h, self.gray, self.red, self.green, self.blue, sf, vh)
        elif self.btnGray.isChecked() == True:
            gavg, gmax, gmin, gstd, histogram = FlashAnalysis(x, y, w, h, self.gray, os.path.splitext(sf)[0] + ".gray.flash.csv")
        elif self.btnRed.isChecked() == True:
            gavg, gmax, gmin, gstd, histogram = FlashAnalysis(x, y, w, h, self.red, os.path.splitext(sf)[0] + ".red.flash.csv")
        elif self.btnGreen.isChecked() == True:
            gavg, gmax, gmin, gstd, histogram = FlashAnalysis(x, y, w, h, self.green, os.path.splitext(sf)[0] + ".green.flash.csv")
        elif self.btnBlue.isChecked() == True:
            gavg, gmax, gmin, gstd, histogram = FlashAnalysis(x, y, w, h, self.gray, os.path.splitext(sf)[0] + ".blue.flash.csv")
            
            
        


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = mainWin()
    my.show()
    sys.exit(app.exec_())
    