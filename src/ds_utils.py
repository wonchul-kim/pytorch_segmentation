import os.path as osp 
from utils.transforms import Compose
import torch 
import torchvision
import utils.helpers as utils
from src.datasets import COCODataset, MaskDataset
from utils.coco_utils import FilterAndRemapCocoCategories, ConvertCocoPolysToMask, _coco_remove_images_without_annotations

def get_dataloader(dataset, dataset_test, args):
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )
    
    return data_loader, data_loader_test, train_sampler

def get_dataset(dir_path, name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)

    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21),
        "mask": (dir_path, get_mask, 21),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes

def get_coco(root, image_set, transforms):
    PATHS = {
        "train": ("train2017", osp.join("annotations", "instances_train2017.json")),
        "val": ("val2017", osp.join("annotations", "instances_val2017.json")),
    }
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

    transforms = Compose([FilterAndRemapCocoCategories(CAT_LIST, remap=True), \
                        ConvertCocoPolysToMask(), transforms])

    img_folder, ann_file = PATHS[image_set]
    img_folder = osp.join(root, img_folder)
    ann_file = osp.join(root, ann_file)

    # dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)
    dataset = COCODataset(img_folder, ann_file, transforms=transforms)

    if image_set == "train": #FIXME: Need to make this option 
        dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset

def get_mask(root, image_set, transforms):
    PATHS = {
        "train": ("train/images"),
        "val": ("val/images"),
    }

    transforms = Compose([transforms])

    img_folder = PATHS[image_set]
    img_folder = osp.join(root, img_folder)

    dataset = MaskDataset(img_folder, image_set, transforms=transforms)

    # if image_set == "train": #FIXME: Need to make this option 
    #     dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset