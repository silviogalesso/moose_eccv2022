import json
import os
import torch
from PIL import Image


class StreetHazards(torch.utils.data.Dataset):
    class_names = ["unlabeled", "building", "fence", "other", "pedestrian", "pole", "road line", "road",
                    "sidewalk", "vegetation", "car", "wall", "trafic sign", "anomaly"]
    num_classes = len(class_names)-1
    ood_indices = [13]
    invalid_indices = []

    def __init__(self, root_folder, split, transforms):
        super(StreetHazards, self).__init__()
        self.root_folder = root_folder
        self.split = split
        with open(os.path.join(root_folder, split, split+".odgt")) as annfile:
            self.data = json.load(annfile)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        imgpath = os.path.join(self.root_folder, self.split, self.data[i]["fpath_img"])
        segmpath = os.path.join(self.root_folder, self.split, self.data[i]["fpath_segm"])
        img = Image.open(imgpath).convert('RGB')
        segm = Image.open(segmpath)
        return self.transforms(img, segm)


