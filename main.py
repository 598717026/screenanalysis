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

# ????????????
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
    #np.savetxt(os.path.splitext(sf)[0] + ".gray.csv", imcutgray, delimiter=',')
    
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
     
        #?????????columns_name
        #writer.writerow(["?????????","?????????","?????????", "?????????"])
        #print([str(gavg), str(gmax), str(gmin), str(gstd)])
        #writer.writerows(["+ %f"%(gavg), "+ %f"%(gmax), "+ %f"%(gmin), "+ %f"%(gstd)])
        
        writer.writerow(["?????????", gavg])
        writer.writerow(["?????????", gmax])
        writer.writerow(["?????????", gmin])
        writer.writerow(["?????????", gstd])
        #???????????????writerows
        writer.writerow(["?????????"])
        writer.writerow(["?????????","??????"])
        writer.writerows(histogram)
    
    return gavg, gmax, gmin, gstd, histogram
    
# MTF??????
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
    
    #????????????
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
    
                
'''????????????QLabel???'''
class myImgLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super(myImgLabel, self).__init__(parent)
        self.hitX = -10
        self.hitY = -10
        self.hitW = 1
        self.hitH = 1
    
    '''??????????????????????????????(??????)'''
    def mousePressEvent(self, event):
        return
        if event.buttons () == QtCore.Qt.LeftButton:                           # ????????????
            #self.setText ("???????????????????????????: ????????????")
            print("??????????????????")  # ??????????????????
        elif event.buttons () == QtCore.Qt.RightButton:                        # ????????????
            self.setText ("???????????????????????????: ????????????")
            print("??????????????????")  # ??????????????????
        elif event.buttons () == QtCore.Qt.MidButton:                          # ????????????
            self.setText ("???????????????????????????: ????????????")
            print("??????????????????")  # ??????????????????
        elif event.buttons () == QtCore.Qt.LeftButton | QtCore.Qt.RightButton: # ?????????????????????
            self.setText ("????????????????????????????????????: ????????????")
            print("?????????????????????")  # ??????????????????
        elif event.buttons () == QtCore.Qt.LeftButton | QtCore.Qt.MidButton:   # ?????????????????????
            self.setText ("????????????????????????????????????: ????????????")
            print("?????????????????????")  # ??????????????????
        elif event.buttons () == QtCore.Qt.MidButton | QtCore.Qt.RightButton:  # ?????????????????????
            self.setText ("????????????????????????????????????: ????????????")
            print("?????????????????????")  # ??????????????????
        elif event.buttons () == QtCore.Qt.LeftButton | QtCore.Qt.MidButton \
             | QtCore.Qt.RightButton:                                          # ????????????????????????
            self.setText ("???????????????????????????????????????: ????????????")
            print("????????????????????????")  # ??????????????????
 
    
    '''??????????????????????????????'''
    def mouseDoubieCiickEvent(self, event):
        if event.buttons () == QtCore.Qt.LeftButton:                           # ????????????
            self.setText ("???????????????????????????: ????????????")
        self.setText ("??????????????????: ????????????(%d, %d)"%(event.x(), event.y()))
 
    '''??????????????????????????????'''
    def mouseCiickEvent(self, event):
        if event.buttons () == QtCore.Qt.LeftButton:                           # ????????????
            self.setText ("???????????????????????????: ????????????")
        self.setText ("??????????????????: ????????????(%d, %d)"%(event.x(), event.y()))
 
    '''?????????????????????????????????'''
    def mouseReleaseEvent(self, event):
        #self.setText("??????????????????: ????????????")
        print("????????????: ????????????(%d, %d)"%(event.x(), event.y()))
        self.hitX = event.x()
        self.hitY = event.y()
        self.update()
        
 
    '''??????????????????????????????'''
    def mouseMoveEvent(self, event):
        #self.setText("??????????????????: ????????????")
        print("????????????")  # ??????????????????
        
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
        self.setWindowTitle("label????????????")

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
        self.spliderW.setMinimum(1)#?????????
        self.spliderW.setMaximum(60)#?????????
        self.spliderW.setSingleStep(1)#??????
        self.spliderW.setTickPosition(QSlider.TicksBelow)#??????????????????????????????
        self.spliderW.setTickInterval(5)#??????????????????
        self.layoutW.addWidget(self.labelW)
        self.layoutW.addWidget(self.spliderW)
        self.widgetW.setLayout(self.layoutW)

        self.widgetH = QWidget()
        self.layoutH = QHBoxLayout()
        self.labelH = QLabel(self)
        self.labelH.setText("hight:" + str(1))
        self.spliderH=QSlider(Qt.Horizontal)
        self.spliderH.valueChanged.connect(self.valChangeH)
        self.spliderH.setMinimum(1)#?????????
        self.spliderH.setMaximum(60)#?????????
        self.spliderH.setSingleStep(1)#??????
        self.spliderH.setTickPosition(QSlider.TicksBelow)#??????????????????????????????
        self.spliderH.setTickInterval(5)#??????????????????
        self.layoutH.addWidget(self.labelH)
        self.layoutH.addWidget(self.spliderH)
        self.widgetH.setLayout(self.layoutH)
        
        
        
        self.widgetSel = QWidget()
        self.layoutSel = QHBoxLayout()
        self.btnW = QRadioButton("????????????")          #??????????????????????????????
        self.btnW.setChecked(True)                     #???????????????????????????
        self.layoutSel.addWidget(self.btnW)     #??????????????????
        self.btnH = QRadioButton("????????????")           #????????????????????????
        self.layoutSel.addWidget(self.btnH)     #???????????????????????????
        self.widgetSel.setLayout(self.layoutSel)   #???????????? layout

        self.btnGray = QRadioButton("Gray????????????") 
        self.layoutSel.addWidget(self.btnGray)     #???????????????????????????
        self.widgetSel.setLayout(self.layoutSel)   #???????????? layout

        self.btnRed = QRadioButton("Red????????????")  
        self.layoutSel.addWidget(self.btnRed)     #???????????????????????????
        self.widgetSel.setLayout(self.layoutSel)   #???????????? layout

        self.btnGreen = QRadioButton("Green????????????") 
        self.layoutSel.addWidget(self.btnGreen)     #???????????????????????????
        self.widgetSel.setLayout(self.layoutSel)   #???????????? layout

        self.btnBlue = QRadioButton("Blue????????????") 
        self.layoutSel.addWidget(self.btnBlue)     #???????????????????????????
        self.widgetSel.setLayout(self.layoutSel)   #???????????? layout

        self.label = myImgLabel(self)
        self.label.setText("   ????????????")
        #self.label.setFixedSize(300, 200)
        #self.label.move(160, 160)

        btn = QPushButton(self)
        btn.setText("????????????")
        btn.clicked.connect(self.openimage)
        
        analysisBtn = QPushButton(self)
        analysisBtn.setText("??????")
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
        imgName, imgType = QFileDialog.getOpenFileName(self, "????????????", "", "*.jpg;*.png;*.bmp;;All Files(*)")
        if imgName == '':
            return
            
        img = QImage(imgName)
        if img.isNull():
            QMessageBox.information(self, '???????????????', '??????????????????%s.' % filename)
            return
        
        self.file_path = imgName
        
        ptr = img.constBits()
        ptr.setsize(img.byteCount())
        self.img = np.array(ptr).reshape( img.height(), img.width(), 4)
        self.gray = 0.299*self.img[:,:,0]+0.587*self.img[:,:,1]+0.114*self.img[:,:,2]
        self.gray = self.gray.astype(np.uint8)
        self.red = self.img[:,:,0]
        self.green = self.img[:,:,1]
        self.blue = self.img[:,:,2]
        
        jpg = QtGui.QPixmap(img)
        print(jpg.width(), jpg.height())
        self.label.setMinimumSize(jpg.width(), jpg.height())
        self.label.setFixedSize(jpg.width(), jpg.height())
        self.label.setPixmap(jpg)
        self.label.resetHitPos()
        self.spliderW.setMaximum(jpg.width() // 2)#?????????
        self.spliderH.setMaximum(jpg.height() // 2)#?????????

    def analysis(self):
        if self.file_path is None:
            return
    
        x, y = self.label.getHitPos()
        w = self.spliderW.value()
        h = self.spliderH.value()
        
        fileName_choose, filetype = QFileDialog.getSaveFileName(self,  
                            "????????????",  
                            os.path.basename(self.file_path) + "." + str(x) + "-" + str(y) + "-" + str(w) + "-" + str(h), # ???????????? 
                            "Csv Files (*.csv)")  
        if fileName_choose == "":
            print("\n????????????")
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
    