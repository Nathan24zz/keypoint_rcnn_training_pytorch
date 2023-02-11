import sys, os
import json
import pydash
import shutil
import numpy as np
import argparse
from tqdm import tqdm

def __parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help="Path to anotation file")
    parser.add_argument("-s", "--source_images_path", help="Path to images directory")
    parser.add_argument("-t", "--target_path", help="Target folder to store anotation")
    parser.add_argument("-n", "--dataset_name", help="Dataset name")
    return parser.parse_args(args)

def xywh2xyxy(bbox, img_size, img_id):
    x1, y1, w, h = bbox
    width, height = img_size
    x2 = x1 + w
    y2 = y1 + h

    if x1 >= width: x1 = width - 1
    if x2 >= width: x2 = width - 1
    if x1 == x2: x2 = x1 + 1
    if y1 >= height: y1 = height - 1
    if y2 >= height: y2 = height - 1
    if y1 == y2: y2 = y1 + 1

    # if x1 >= width or x2 >= width or x1 == x2: print(img_id)
    # if y1 >= width or y2 >= width or y1 == y2:print(img_id)
    return [x1, y1, x2, y2]

def get_keypoint(keypoints, img_size):
    width, height = img_size

    kpts = []
    for i in range(0, len(keypoints), 3):
        x,y,v = keypoints[i:i+3]
        if x >= width: x = width - 1
        if y >= height: y = height - 1

        # if x >= width: print('x')
        # if y >= height: print('y')
        kpts.append([x,y,v])
    return kpts


if __name__ == '__main__':
    """
    Script to convert coco format anotation to yolo format.
    Input :
    -i = Path to anotation file
    -s = Path to images directory
    -t = Target folder to store anotation
    -n = Dataset name
    Example : python coco2yolo.py -i '/mnt/d/Reza/Dokumen/datasets/robot_pose/robot_keypoints_train_fixed.json' -s '/mnt/d/Reza/Dokumen/datasets/robot_pose/images/' -t "yolo_robot_pose" -n "train"
              python coco2yolo.py -i '/mnt/e/kuliah/Semester_7/PRA_TA/HumanoidRobotPoseEstimation/dataset/hrp/robot_keypoints_test_fixed.json' -s '/mnt/e/kuliah/Semester_7/PRA_TA/HumanoidRobotPoseEstimation/dataset/hrp/images/' -t 'yolo_robot_pose_1' -n 'test'
    """
    arguments = __parse_arguments(sys.argv[1:])
    input_file = arguments.input_file
    source_images_path = arguments.source_images_path
    target_path = arguments.target_path
    dataset_name = arguments.dataset_name

    # make target directory
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # load coco file
    data = json.load(open(input_file) )

    # make images folder
    target_images_path = os.path.join(target_path, dataset_name, "images")
    if os.path.exists(target_images_path):
        shutil.rmtree(target_images_path, ignore_errors=True)
    os.makedirs(target_images_path)

    # make annotations folder
    target_anotation_path = os.path.join(target_path, dataset_name, "annotations")
    if os.path.exists(target_anotation_path):
        shutil.rmtree(target_anotation_path, ignore_errors=True)
    os.makedirs(target_anotation_path)

    # group annotations based on image_id
    annotations = pydash.group_by(data['annotations'], 'image_id')

    annotation_content = {
        "bboxes": [],
        "keypoints": []
    }

    for image in tqdm(data['images']):
        file_name = image['file_name']
        file_name_no_ext = file_name.split('.')[0]
        image_id = image['id']

        # copy images to target dir
        src = os.path.join(source_images_path, file_name)
        dst = os.path.join(target_images_path, file_name)
        shutil.copyfile(src, dst)

        annotations_data = annotations[image_id]
        height = image['height']
        width = image['width']
        for anotation in annotations_data:
            annotation_content['bboxes'].append(xywh2xyxy(anotation['bbox'], [width, height], image_id))
            annotation_content["keypoints"].append(get_keypoint(anotation['keypoints'], [width, height]))

        with open(os.path.join(target_anotation_path, file_name_no_ext + ".json"), 'w') as outfile:
            json.dump(annotation_content, outfile)
        
        annotation_content['bboxes'] = []
        annotation_content['keypoints'] = []
