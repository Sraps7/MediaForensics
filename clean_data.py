r"""本module用于对dfdc数据集进行清洗工作，具体如下：

作用对象：通过mtcnn得到的视频中不同帧中人脸boxes位置的json文件

作用：
1.去除较小的box
2.将box大小扩大1.5倍

author: Yang Piaoyang
date: 2020/07/10
"""
import os
import json
import logging
logging.basicConfig(filename='clean-log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# dirs = ['/home/data/dfdc/dfdc_train/dfdc_train_part_{}/'.format(x) for x in range(50)]
metas = ['/home/data/dfdc/dfdc_train/dfdc_train_part_{}/metadata.json'.format(x) for x in range(50)]


def rm_small_area(content: dict) -> dict:
    for key in content.keys():
        boxes = content[key]
        if boxes == []:
            continue
        idxes_for_rm = []
        for idx, box in enumerate(boxes):
            if box[2] - box[0] < 70 or box[3] - box[1] < 70:
                idxes_for_rm.append(idx)
        count = 0
        for idx in idxes_for_rm:
            idx = idx - count
            content[key].pop(idx)
            count = count + 1
    return content


def expand_size(content: dict) -> dict:
    for key in content.keys():
        boxes = content[key]
        if boxes == []:
            continue
        for idx, box in enumerate(boxes):
            center = (0.5 * (box[2] + box[0]), 0.5 * (box[3] + box[1]))
            half_edge = 1.5 * max(center[0] - box[0], center[1] - box[1])
            logging.debug("{}: {}: centre: {}; half_edge:{}".format(key, idx, center, half_edge))
            content[key][idx][0] = max(0, center[0] - half_edge)
            content[key][idx][1] = max(0, center[1] - half_edge)

            # 由于视频有1920*1080和1080*1920两种格式，所以在限制box中第二个点的坐标时只能做如下的粗略约束
            content[key][idx][2] = min(1920, center[0] + half_edge)
            content[key][idx][3] = min(1920, center[1] + half_edge)
    return content


def main():
    for idx, ameta in enumerate(metas):
        with open(ameta, 'r', encoding='utf-8') as reader:
            metadata = json.load(reader)
        for iidx, key in enumerate(metadata.keys()):
            if metadata[key]['label'] == 'REAL':
                dirname = os.path.dirname(ameta)
                filename = os.path.join(dirname, key.replace('mp4', 'json'))
                with open(filename, 'r', encoding='utf-8') as reader:
                    content = json.load(reader)

                content = rm_small_area(content)
                content = expand_size(content)

                filename_clean = os.path.join(dirname, "clean-" + key.replace('mp4', 'json'))
                with open(filename_clean, 'w', encoding='utf-8') as writer:
                    json.dump(content, writer)
                if iidx % 10 == 0:
                    logging.debug("video {}: {}: {} is over".format(idx, iidx, key))
        
        logging.info("dir {}: {} is over".format(idx, os.path.dirname(ameta)))
    logging.info("all is over")


def test_rm_small_area():
    path = "../kaggle/sample/aahncigwte.json"
    with open(path, 'r', encoding='utf-8') as reader:
        content = json.load(reader)
    content = rm_small_area(content)
    dirname = os.path.dirname(path)
    key = os.path.basename(path).replace('json', 'mp4')
    filename_clean = os.path.join(dirname, "clean-" + key.replace('mp4', 'json'))
    with open(filename_clean, 'w', encoding='utf-8') as writer:
        json.dump(content, writer)

def test_expand_size():
    path = "../kaggle/sample/aahncigwte.json"
    with open(path, 'r', encoding='utf-8') as reader:
        content = json.load(reader)
    content = expand_size(content)
    dirname = os.path.dirname(path)
    key = os.path.basename(path).replace('json', 'mp4')
    filename_clean = os.path.join(dirname, "clean-" + key.replace('mp4', 'json'))
    with open(filename_clean, 'w', encoding='utf-8') as writer:
        json.dump(content, writer)

if __name__ == "__main__":
    main()
    # test_rm_small_area()
    # test_expand_size()


        
