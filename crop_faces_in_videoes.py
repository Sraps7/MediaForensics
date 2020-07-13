r"""
这个module用来将dfdc数据集中的视频中的人脸部分crop出来，保存到文件（.jpg）

本模块的使用前提：需要提供人脸区域的boxes坐标

采用多进程
"""

import cv2
import os
import json
import logging
import numpy as np
logging.basicConfig(filename='crop-log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import time
from multiprocessing import Pool, Value, Queue
# from pudb.remote import set_trace

DEBUG = True
metas = ['/home/data/dfdc/dfdc_train/dfdc_train_part_{}/metadata.json'.format(x) for x in range(50)]
dirty_img_count = 0

def img_diff(img, ori_img) -> bool:
    global dirty_img_count
    diff = np.mean(np.abs(img - ori_img) > 10)
    logging.debug("diff: {}".format(diff))
    if diff > 0.1:
        return True
    dirty_img_count += 1
    return False


def crop_faces(idx: int, avideo: str, label: bool, ori_video="") -> None:
    # set_trace()
    try:
        dirname = os.path.dirname(metas[idx])
        dest_dir = "/home/data/dfdc/faces/train/"
        if label == "REAL":
            json_file = os.path.join(dirname, "clean-" + avideo.replace("mp4", "json"))
        elif ori_video != "":
            json_file = os.path.join(dirname, "clean-" + ori_video.replace("mp4", "json"))
        else:
            raise ValueError('if `label` == False, `ori_video` must be provided')

        logging.debug("json_file: {}".format(json_file))

        video_file = os.path.join(dirname, avideo)
        with open(json_file, 'r', encoding='utf-8') as reader:
            frames = json.load(reader) # dict
        
        cap = cv2.VideoCapture(video_file)

        if label == 'REAL':
            idx = 0
            while True:
                sign, img = cap.read()
                if sign:
                    idx += 1
                    logging.debug("frames.kyes: {}".format(frames.keys()))
                    if str(idx) in frames.keys():
                        if frames[str(idx)] == []:
                            continue
                        else:
                            boxes = frames[str(idx)]
                            logging.debug("boxes: {}".format(boxes))
                            for box in boxes:
                                crop = img[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                                filename = os.path.join(dest_dir + "0/", avideo.split('.')[0] + '-' + str(idx) + '.jpg')
                                logging.debug('filename: {}'.format(filename))
                                retval = cv2.imwrite(filename, crop)
                                if not retval:
                                    raise ValueError("cv2 write img {} error".format(filename))
                else:
                    break
        else:
            ori_video_file = os.path.join(dirname, ori_video)
            cap_ori = cv2.VideoCapture(ori_video_file)
            idx = 0
            while True:
                sign, img = cap.read()
                sign_ori, img_ori = cap_ori.read()
                if sign and sign_ori:
                    idx += 1
                    if str(idx) in frames.keys():
                        if frames[str(idx)] == []:
                            continue
                        else:
                            boxes = frames[str(idx)]
                            for box in boxes:
                                crop = img[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                                crop_ori = img_ori[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                                if not img_diff(crop, crop_ori):
                                    logging.debug("no diff")
                                    continue

                                filename = os.path.join(dest_dir + "1/", avideo.split('.')[0] + '-' + str(idx) + '.jpg')
                                logging.debug('filenaem: {}'.format(filename))
                                retval = cv2.imwrite(filename, crop)
                                if not retval:
                                    raise ValueError("cv2 write img {} error".format(filename))
                else:
                    break
        logging.info("this sub is over {}".format(os.getpid()))
    except Exception as e:
        raise ValueError("value erro: {}".format(e))
        logging.debug("run with exceptrion: {}".format(e))


def main():
    for idx, ameta in enumerate(metas):
        with open(ameta, 'r', encoding='utf-8') as reader:
            videos = json.load(reader)
        
        p = Pool(8)
        num_videos = len(videos.keys())
        for iidx, avideo in enumerate(videos.keys()):
            logging.debug('iidx: {}'.format(iidx))
            label = videos[avideo]["label"]
            if label == "REAL":
                ori_video = ""
            else:
                ori_video = videos[avideo]["original"]
            p.apply_async(crop_faces, (idx, avideo, label, ori_video))

            # if label == "REAL":
            #     logging.debug("{}".format(videos[avideo]))
            #     crop_faces(idx, avideo, label)
            # else:
            #     ori_video = videos[avideo]["original"]
            #     crop_faces(idx, avideo, label, ori_video)
            # sub_exception = q_share.get()
            # logging.debug('sub_exception: {}'.format(sub_exception))
            if DEBUG:
                if iidx > 19:
                    p.close()
                    p.join()
                    return
            logging.debug("video {}: {}: {}: {} is over".format(idx, iidx, os.path.dirname(ameta), avideo))
            if iidx % 50 == 0:
                logging.info("video {}/{}: dir {} is over".format(iidx, num_videos-1, idx))
        p.close()
        p.join()

        logging.info("dir {}: {} is over".format(idx, os.path.dirname(ameta)))
    logging.info("all is over")
    logging.info("rm {} dirty imgs".format(dirty_img_count))


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    logging.info('time: {}'.format(end - start))