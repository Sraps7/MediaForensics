#coding:utf-8
import sys
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
import cv2
import os
import numpy as np
import time
from multiprocessing import Pool, Value
import json
# from pudb.remote import set_trace

test_mode = "ONet"
thresh = [0.6, 0.7, 0.7]
min_face_size = 20
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]
batch_size = [2048, 64, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

DEBUG = True

if slide_window:
    PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
else:
    PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet

# load rnet model
if test_mode in ["RNet", "ONet"]:
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = RNet

# load onet model
if test_mode == "ONet":
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                            stride=stride, threshold=thresh, slide_window=slide_window)


def extract_face(avideo_path, num_share, total_share):
    # set_trace(term_size=(80, 24))
    # print('run task %s (%s)...' % (avideo_path, os.getpid()))
    # if slide_window:
    #     PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
    # else:
    #     PNet = FcnDetector(P_Net, model_path[0])
    # detectors[0] = PNet

    # # load rnet model
    # if test_mode in ["RNet", "ONet"]:
    #     RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    #     detectors[1] = RNet

    # # load onet model
    # if test_mode == "ONet":
    #     ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    #     detectors[2] = ONet

    # mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
    #                             stride=stride, threshold=thresh, slide_window=slide_window)

    video_capture = cv2.VideoCapture(avideo_path)
    count = 0

    boxes_json = avideo_path.replace('mp4', 'json')
    boxes_dict = {} # {frame(int): boxes(np.ndarry)}
    while True:
        sign, frame = video_capture.read()
        count = count + 1
        if sign and count % 20 == 1:
            boxes, landmarks = mtcnn_detector.detect(frame)
            boxes_lst = boxes.tolist()
            boxes_dict[count] = boxes_lst
       
        elif not sign:
            break
            
    
    video_capture.release()
    try:
        with open(boxes_json, 'w', encoding='utf-8') as writer:
            json.dump(boxes_dict, writer)
            # num_share.value = num_share.value + 1
            num_share = num_share + 1
    except:
        print('writer error, obj_path: ', boxes_json)
    
    # if num_share.value % 10 == 1:
    #     print("num_share(finished): {} / {}".format(num_share.value, total_share.value))
    if num_share % 10 == 1:
        print("num_share(finished): {} / {}".format(num_share, total_share))
    
    return num_share
    # print('return task %s (%s)...' % (avideo_path, os.getpid()))


if __name__ == "__main__":
    
   
    path = "/home/data/dfdc/dfdc_train/"
    # path = "e:/kaggle/dfdc_train/"
    # path = './lala/'
    
    dirs = []
    for dirpaths, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dirs.append(os.path.join(dirpaths, dirname))


    for idx, adir in enumerate(dirs):
        if DEBUG:
            if idx > 0:
                break

        print('now dir: ', adir.split('/')[-1])

        # videos_paths = []
        # for afile in os.listdir(adir):
        #     if afile.split('.')[-1] == 'mp4':
        #         videos_paths.append(os.path.join(adir, afile))
        
        videos_json = os.path.join(adir, 'metadata.json')
        
        real_video_adre = []

        with open(videos_json, 'r', encoding='utf-8') as reader:
            jsons = json.load(reader)
        for akey in jsons.keys():
            label = jsons[akey]['label']
            if label == 'REAL':
                real_video_adre.append(os.path.join(adir, akey))
    
        len_reals = len(real_video_adre)
        
        # num_share = Value('d', 0)
        # total_share = Value('d', len_reals)
        num_share = 0
        total_share = len_reals
        # p = Pool(4)
        for areal_video in real_video_adre:
        #     p.apply_async(extract_face, (areal_video, num_share, total_share))
            num_share = extract_face(areal_video, num_share, total_share)


        print("now, finish! idx: {}, dir: {}.".format(idx, adir.split('/')[-1]))
