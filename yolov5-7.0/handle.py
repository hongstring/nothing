import math
import sys
import threading
import time
import numpy as np
import pynput
import torch,win32api
import pyautogui
import win32con

import win32gui
from PIL import ImageGrab
import ctypes
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (cv2, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator
from utils.torch_utils import select_device, smart_inference_mode
from corefn import screenshot,Dynamic_AttackRange
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController


weights = 'best.pt'
device = select_device("cuda:0")
model = DetectMultiBackend(weights, device=device, dnn=False, data=False, fp16=True)
model.eval()

names = model.names
#获取屏幕中心点
screen_width = win32api.GetSystemMetrics(0)
screen_height = win32api.GetSystemMetrics(1)
CoreX = int(screen_width / 2)
CoreY = int(screen_height / 2)
# 设置按键延迟时间
pyautogui.PAUSE = 1
#print (mode1)
# 获取英雄联盟窗口的句柄
hwnd = win32gui.FindWindow(None, "League of Legends (TM) Client")
# # 设置窗口位置和大小
win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 0, 0, 1280, 720, win32con.SWP_SHOWWINDOW)
#读取图片
last_yita_pos = (1222, 707)
keyboard = KeyboardController()
mouse = MouseController()
gth_pos = None
shenji_pos  = None
while True:
    # im0 = screenshot()
    bbox = (0, 0, 1280, 720)
    im0 = np.array(ImageGrab.grab(bbox=bbox))
    src_shape = im0.shape
    #处理图片
    im = letterbox(im0, (1280, 1280), stride=32, auto=True)[0] # padded resize
    im = im.transpose((2, 0, 1))[::-1] # HNC to CHN, BGR to RGB
    im = np.ascontiguousarray(im) # contiguous
    im = torch.from_numpy(im).to(model .device)
    im = im.half() if model.fp16 else im.float() # uint8 to fp16/32
    im /= 255#0-255to0.0一1.0
    if len(im.shape) == 3:
        im = im[None] # expand for batch dim
    #推理
    pred = model(im, augment=False, visualize=False)
    #非极大值抑制
    pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.45, classes=None, max_det=200)[0]
   # pred = non_ max_ suppression(pred, conf_ thres=0.6; iou_ thres=0.45， classes=0, max. det=1000)
  # 区别在于[8]: tensor([], device= 'cuda:0', size=(0, 6)) 与tensor([], device=' cuda:0', size=(0, 6)
  # 就是少了个[]

    if not len(pred):
         print( "未检测到目标")
    else:
        conf_list = []
        target_list = []
        product_list = []
        label_list = []
        index_list = []  # 用来记录添加到列表中的元素的索引

        for *xyxy, score, label in reversed(pred):  # 处理推理出来每个目标的信息
            # print(xyxy)
            # 用map函数将x1, y1, x2,y2转换为round类型
            x1, y1, x2, y2 = map(round, (torch.tensor(xyxy).view(1, 4).view(-1).tolist()))
            x, y, w, h = map(round, ((xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()))
            score = float(score)
            # im0 = Dynamic_AttackRange(x1, y1, x2, y2, x, y, im0)
            if label.item() in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21]:
                # 计算距离
                dis = math.sqrt(math.pow(x - CoreX, 2) + math.pow(y - CoreY, 2))
                conf_list.append(score)
                target_list.append([x, y, w, h])
                product_list.append(w * h)
                label_list.append(label.item())
                # 记录添加到列表中的元素的索引
                index_list.append(len(conf_list) - 1)



        # 处理 label_list 的长度
        if len(label_list) < len(index_list):
            label_list.extend([None] * (len(index_list) - len(label_list)))

        if len(conf_list) == 0:
            print("未检测到目标")
        else:
            # 循环每个检测到的目标
            for *xyxy, score, label in reversed(pred):
                x1, y1, x2, y2 = map(round, (torch.tensor(xyxy).view(1, 4).view(-1).tolist()))
                current_image_width = 1280
                original_image_width = 1280
                current_image_height = 720
                original_image_height = 720
                w_ratio = current_image_width / original_image_width
                h_ratio = current_image_height / original_image_height
                # 在图片上画框
                x1 = int(x1 * w_ratio)
                y1 = int(y1 * h_ratio)
                x2 = int(x2 * w_ratio)
                y2 = int(y2 * h_ratio)
                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 在框上标注标签
                cv2.putText(im0, str(label.item()), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            for i in range(len(index_list)):
                j = index_list[i]
                x, y, W, h = target_list[j]
                print("目标信号坐标:", x, y, W, h, "标签为：", label_list[j] if label_list[j] is not None else "N/A")

                #循环遍历，找到第一个名字叫GTH的，获取它的坐标
                for m in range(len(index_list)):
                    if names.get(label_list[m], "") == "GTH":
                        gth_pos = target_list[m]
                        print("感叹号的位置在:", gth_pos if gth_pos is not None else "N/A")
                        break
                for sj in range(len(index_list)):
                    if names.get(label_list[sj], "") == "Shenji":
                        shenji_pos = target_list[sj]
                        print("升级的位置在:", shenji_pos if shenji_pos is not None else "N/A")
                        break
                # 英雄联盟脚本
                # 开局记录标签为一塔的坐标,并永不更改
                # if label_list[j] == 14 and last_yita_pos is None:
                #     last_yita_pos = (x,y)
                #     # 如果检测到在标签为泉水，则打开按下背包键P键，双击标签为GTH的坐标，双击该坐标位置以购买物品，5秒后按下P键关闭背包，1分钟后右键前往一塔，期间停止检测10秒
                #     # 打开背包
                if label_list[j] == 2:
                    # 升级技能辣
                    pyautogui.moveTo(shenji_pos[0], shenji_pos[1])
                    time.sleep(1)
                    mouse.click(Button.left)
                    mouse.release(Button.left)
                if label_list[j] == 12:
                    #按P打开背包
                    keyboard.press('p')
                    keyboard.release('p')
                    if gth_pos is not None:
                        pyautogui.moveTo(gth_pos[0], gth_pos[1])
                        time.sleep(1)
                        mouse.click(Button.left, clicks=2)
                        mouse.release(Button.left)
                        #再次按P关闭背包
                        keyboard.press('p')
                        keyboard.release('p')
                        time.sleep(1)

                        mouse.press(Button.left,2)
                        mouse.release(Button.left)
                        time.sleep(5)
                        mouse.press(Button.left, 2)
                        mouse.release(Button.left)
                    pyautogui.moveTo(last_yita_pos[0], last_yita_pos[1])
                    time.sleep(30)

                # 敌我小兵差距
                difference_value = label_list.count(0) - label_list.count(1)
                if label_list.count(1) == 1 and difference_value >= 5:
                    # 右键一塔的坐标，回到一塔
                    pyautogui.rightClick(x=last_yita_pos[0], y=last_yita_pos[1])
                    print("敌方人多，撤退")
                    time.sleep(5)
                else:
                    # 按下A键后左键某一个记录值为友方小兵的坐标以此跟随我方小兵，并自动攻击，停止检测2秒
                    friend_xb_pos = None
                    enemy_xb_pos = None
                    # for k in range(len(index_list)):
                    #     if names.get(label_list[k],"") == "WFXB":
                    #         friend_xb_pos = target_list[k]
                    #         break
                    for k in range(len(index_list)):
                        if names.get(label_list[k], "") == "DFXB":
                            friend_xb_pos = target_list[k]
                            break
                    if friend_xb_pos is not None:
                        # 创建键盘和鼠标控制器对象

                        keyboard.press('a')
                        keyboard.release('a')
                        time.sleep(0.1)
                        pyautogui.moveTo(friend_xb_pos[0],friend_xb_pos[1])
                        time.sleep(0.1)
                        mouse.press(Button.right)
                        mouse.release(Button.right)
                        friend_xb_pos = None
                        print("开始跟随小兵，攻击")
                        time.sleep(6)
