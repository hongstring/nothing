import mss,cv2,win32api,os,time
import numpy as np
from pynput import mouse,keyboard
from threading import Thread
from multiprocessing import Manager


sct = mss.mss()
def screenshot():
    leftx = int((win32api.GetSystemMetrics(0) / 2) - 960)  # 左上角x坐标
    topy = int((win32api.GetSystemMetrics(1) / 2) - 640)  # 左上角y坐标

    monitor = {"top": topy, "left": leftx, "width": 1920, "height": 1080}

    # 获取屏幕图像
    img = sct.grab(monitor)
    img = np.array(img)
    img = np.array(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)

    return img

def Dynamic_AttackRange(x1,y1,x2,y2,x,y,img):
    img=cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),1)
    img=cv2.circle(img,(round(x),round(y)),2,(255,255,255),1)
    return img