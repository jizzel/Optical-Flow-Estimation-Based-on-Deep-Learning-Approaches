from pathlib import Path
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from enhance_image import elaborateImage

BLUE = [255, 0, 0]
img1 = cv.imread('first.png')
elaborateImage(img1)
# replicate = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_REPLICATE)
# reflect = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_REFLECT)
# reflect101 = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_REFLECT_101)
# wrap = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_WRAP)
# constant = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=BLUE)
# constant[:, 2, :] = 0
# plt.subplot(231), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')
# plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
# plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
# plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
# plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
# plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
# plt.show()
# import PIL

# from PIL import Image

# im = Image.open("first.png")
# im.rotate(145).show()

# creating gif image object
# img = Image.open("time.gif")
# img1 = img.tell()
# print(img1)
#
# # using seek() method
# img2 = img.seek(img.tell() + 1)
# img3 = img2.tell()
# print(img3)
# img.show()

# video_reader = cv2.VideoCapture('./data/train.mp4')
# # print('video path: ', video_input_path)
# print('video reader: ', video_reader)
#
# num_frames = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
# frame_size = (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# fps = int(video_reader.get(cv2.CAP_PROP_FPS))
# fourcc = 0x00000021
# video_writer = cv2.VideoWriter('./experiment', fourcc, fps, frame_size)
# video_writer.write('./first.png')
# path = Path('out.flo')
# with path.open(mode='r') as flo:
#     np_flow = np.fromfile(flo, np.float32)
#     print(np_flow.shape)
#
# with path.open(mode='r') as flo:
#   tag = np.fromfile(flo, np.float32, count=1)[0]
#   width = np.fromfile(flo, np.int32, count=1)[0]
#   height = np.fromfile(flo, np.int32, count=1)[0]
#   print('test ', np.fromfile(flo, np.float32, count=1)[0])
#   print('test2 ', np.fromfile(flo, np.float32, count=1)[0])
#   print('test1 ', np.fromfile(flo, np.float32, count=1)[0])
#   print('test12 ', np.fromfile(flo, np.float32, count=1)[0])
#
#   print('tag', tag, 'width', width, 'height', height, '\n')
#
#   nbands = 2
#   tmp = np.fromfile(flo, np.float32, count= nbands * width * height)
#   flow = np.resize(tmp, (int(height), int(width), int(nbands)))
#
# print('flow: ', flow[0][1])
# print('flow shape: ', flow.shape)

# a='/images/frame_000'
# b='.png'
# output='out.flo'
# for i in range(50):
#   python run.py --model sintel-final --first a+i+b --second ./images/second.png --out ./output
#   i=i+2
# done
