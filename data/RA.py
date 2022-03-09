import json
import os
import torch
from PIL import Image


class RoadAnomaly(torch.utils.data.Dataset):
    num_classes = 19
    ood_indices = [2]
    invalid_indices = []

    def __init__(self, root_folder, transforms):
        super(RoadAnomaly, self).__init__()
        self.root_folder = root_folder
        self.data = RoadAnomaly._get_data(root_folder)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _get_data(root_folder):
        with open(os.path.join(root_folder, "frame_list.json")) as fl:
            frames_list = json.load(fl)
        data = []
        for frame in frames_list:
            data.append({"fpath_img": os.path.join(root_folder, "frames", frame),
                         "fpath_segm": os.path.join(root_folder, "frames", frame.replace("jpg", "labels"), "labels_semantic.png")})
            assert os.path.isfile(data[-1]["fpath_img"]), data[-1]["fpath_img"]
            assert os.path.isfile(data[-1]["fpath_segm"]), data[-1]["fpath_segm"]
        return data

    def __getitem__(self, i):
        imgpath = self.data[i]["fpath_img"]
        segmpath = self.data[i]["fpath_segm"]
        img = Image.open(imgpath).convert('RGB')
        segm = Image.open(segmpath)
        return self.transforms(img, segm)


