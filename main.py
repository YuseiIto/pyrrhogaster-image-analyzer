#coding:utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2 
import csv
import os



# User Configurations
SOURCE_LIST_CSV='index.csv'
SOURCE_DIR='./source'
SOURCE_ENCODING="UTF-8"

DEST_CSV="dest.csv"
DEST_ENCODING="UTF-8"


LABELS={
  'id':'ID',
  'pixels_sum':'合計ピクセル',
  'red_pixels':'赤ピクセル',
  'black_pixels':'黒ピクセル',
  'red_islands': '赤独立数',
  'black_islands': '黒独立数',
}

# Image process Configurations
COLOR_THRESHOLD=60
CONTOURS_AREA_MIN_THRESH=20
USE_BLUR=False


# Other preferences
CONTOURS_COLOR=(0,255,0)

# --------------- CODE  --------------- 

# Image Process

def readImage(path): # path-> tuple(img_color,img_bin,img_bin_inv)
  img_color = cv2.imread(path)
  img_gray = cv2.imread(path,0)

  thresh_source=img_gray

  if USE_BLUR:
    img_blur = cv2.blur(img_gray,(3,3)) #3*3領域で平滑
    thresh_source=img_blur
  ret, img_bin = cv2.threshold(thresh_source, COLOR_THRESHOLD, 255, cv2.THRESH_BINARY)

  ret_inv, img_bin_inv = cv2.threshold(thresh_source, COLOR_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
  ret_mask,mask = cv2.threshold(thresh_source,200, 255, cv2.THRESH_BINARY_INV) # 必要な部分が白(255)、不要な部分が黒(0)
  img_bin_inv[np.where(mask==[0])] = [255]
  return (img_color,img_bin,img_bin_inv)



def detectContours(img_bin): # img -> contours
  contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  contours = list(filter(lambda x: cv2.contourArea(x) > CONTOURS_AREA_MIN_THRESH, contours))
  return contours

# Main routine
def analyseImage(image_path):
  (img_color,img_bin,img_bin_inv)=readImage(image_path)
  contours=detectContours(img_bin) # black contours
  contours_inv=detectContours(img_bin_inv) # red/yellow contours
  black_islands=len(contours)
  red_islands=len(contours_inv)
  black_pixels=np.count_nonzero(img_bin == 0)
  red_pixels=np.count_nonzero(img_bin_inv == 0)
  pixels_sum=black_pixels+red_pixels
  return (pixels_sum,red_pixels,black_pixels,red_islands,black_islands)



# Window utils

def showImageUntilKey(img):
  cv2.imshow("window",img)
  cv2.waitKey()

def showContours(contours,base_image):
  img=np.copy(base_image)
  cv2.drawContours(img, contours, -1, CONTOURS_COLOR, 2)
  showImageUntilKey(img)



# file utils
def readCSV(path,encoding): # path -> list[list[...],...]
  fs=open(path, encoding=encoding)
  reader =csv.reader(fs)
  rows = [row for row in reader]
  fs.close()
  return rows


def writeCSV(path,encoding,data): #path: string , data: list[list[...],...]
 lines=[]
 for row in data:
   lines.append(",".join(map(str,row)))
 csv_string="\n".join(lines)
 fd = open(path, 'w',encoding=encoding)
 fd.write(csv_string)
 fd.close()

# Route routine

rows=readCSV(SOURCE_LIST_CSV,SOURCE_ENCODING)
label_row=[LABELS['id'],LABELS['pixels_sum'],LABELS['red_pixels'],LABELS['black_pixels'],LABELS['red_islands'],LABELS['black_islands']]
results=[label_row] # [[id, pixels_sum, red_pixels, black_pixels,red_islands,black_islands], ...]

for row in rows[1:]:
 [id,file_name]=row
 (pixels_sum,red_pixels,black_pixels,red_islands,black_islands)=analyseImage(os.path.join(SOURCE_DIR,file_name))
 results.append([id,pixels_sum,red_pixels,black_pixels,red_islands,black_islands])


writeCSV(DEST_CSV,DEST_ENCODING,results)
