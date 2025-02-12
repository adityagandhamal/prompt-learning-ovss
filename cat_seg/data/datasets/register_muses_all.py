import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

import copy
 

def muses():
    muses_classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


    ret = {
        "stuff_classes" : muses_classes,
    }
    return ret

def register_muses_all(root):
    root = os.path.join(root, "MUSES_ALL")
    #print('root---',root)
    meta = muses()
    #print('meta---',meta)
    #print(hey)
    for image_dirname, sem_seg_dirname in [
        ("muses_imgs", "annotations_detectron2"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        #print('image_dir--',image_dir)
        gt_dir = os.path.join(root, sem_seg_dirname)
        #print('gt_dir--',gt_dir)
        #print(hey)
        name = "MUSES_ALL_CONDITIONS"
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='png'))
        MetadataCatalog.get(name).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=255, **meta,)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_muses_all(_root)
#if dataset_name in DatasetCatalog.list():
#print('dataset_name---',DatasetCatalog.list())
#print('done cityscapes !!!!')



#["unlabeled", "ego vehicle","rectification border","out of roi", "static", "dynamic",  "ground" , "road" , "sidewalk", "parking", "rail track", "building", "wall", "fence", "guard rail", "bridge", "tunnel","pole", "polegroup",   "traffic light",  "traffic sign"  ,"vegetation" ,   "terrain" ,   "sky" , "person" ,  "rider","car","truck" ,"bus",  "caravan" ,"trailer", "train" ,"motorcycle","bicycle"]

#cityscapes_classes = ["ego vehicle","rectification border","out of roi", "static", "dynamic",  "ground" , "road" , "sidewalk", "parking", "rail track", "building", "wall", "fence", "guard rail", "bridge", "tunnel","pole", "polegroup",   "traffic light",  "traffic sign"  ,"vegetation" ,   "terrain" ,   "sky" , "person" ,  "rider","car","truck" ,"bus",  "caravan" ,"trailer", "train" ,"motorcycle","bicycle"]


