import os
import re
import argparse
from collections import OrderedDict
from tqdm import tqdm
from pprint import pprint
# import matplotlib.pyplot as plt
from time import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import StreetHazards, RoadAnomaly, transforms as tr
from utils import aupr, fpr_at_tpr, AverageValueMeter
from models.moose import deeplabv3_resnet50_moose, deeplabv3_resnet101_moose, deeplabv3plus_resnet101_moose
from utils.ood_score_functions import ood_score_functions


def main(args):
    # DATA
    if args.dataset == "street_hazards":
        print(">> Loading StreetHazards dataset")
        test_transforms = tr.Compose([tr.ToTensor(), tr.LabelOffset(offset=1)])
        test_data = StreetHazards(os.path.join(args.datasets_folder, "street_hazards"), "test", test_transforms)
    elif args.dataset == "road_anomaly":
        print(">> Loading RoadAnomaly dataset")
        test_transforms = tr.Compose([tr.ToTensor(), tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_data = RoadAnomaly(os.path.join(args.datasets_folder, "RoadAnomaly_jpg"), test_transforms)
    else:
        raise NotImplementedError

    test_loader = DataLoader(test_data, shuffle=False, batch_size=1, num_workers=args.num_workers)

    # MODEL
    if args.arch == "deeplabv3_resnet50":
        model = deeplabv3_resnet50_moose(num_classes=test_data.num_classes,
                                         probes_depth=args.probes_depth).cuda()
    elif args.arch == "deeplabv3_resnet101":
        model = deeplabv3_resnet101_moose(num_classes=test_data.num_classes,
                                          probes_depth=args.probes_depth).cuda()
    elif args.arch == "deeplabv3plus_resnet101":
        model = deeplabv3plus_resnet101_moose(num_classes=test_data.num_classes,
                                              output_stride=16,
                                              probes_depth=args.probes_depth).cuda()
    else:
        raise NotImplementedError
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.checkpoint_path)['model_state_dict'])

    print(">> total parameters = {:.03f}M".format(sum([p.numel() for p in model.parameters()]) * 1e-6))

    eval_ens_ood(model, test_loader)


def eval_ens_ood(model, loader):
    model.eval()
    metrics = OrderedDict([
        (score_fn, OrderedDict([
            ("AUPR", {"function": lambda pr, gt: 100*aupr(pr.view(-1), gt.view(-1)),
                      "meter": AverageValueMeter(),
                      "meter_moose": AverageValueMeter()}),
            ("FPR@95TPR", {"function": lambda pr, gt: 100 * fpr_at_tpr(pr.view(-1), gt.view(-1), tpr=0.95),
                           "meter": AverageValueMeter(),
                           "meter_moose": AverageValueMeter()}),
        ])) for score_fn in ood_score_functions.keys()])
    runtime_meter = AverageValueMeter()

    t = tqdm(total=len(loader.dataset)//loader.batch_size)
    for idx, (img, segm) in enumerate(loader):
        img = img.cuda()
        # prepare ground truth OoD segmentation
        ood_segm = (segm == loader.dataset.ood_indices[0])
        for label in loader.dataset.ood_indices[1:]:
            ood_segm = torch.logical_or(ood_segm, segm==label)
        ood_segm = ood_segm.float()
        if ood_segm.sum() == 0: continue
        for label in loader.dataset.invalid_indices:
            ood_segm[segm==label] = 255
        start = time()
        with torch.no_grad():
            model_output = model(img)
        runtime_meter.add(time() - start)
        pred = model_output['out']
        all_preds = model_output["contextual_out"] + [pred]

        # main prediction
        for score_fn in ood_score_functions.keys():
            ood_scores_global = ood_score_functions[score_fn]["single"](pred)
            for key, metric in metrics[score_fn].items():
                val = metric["function"](ood_scores_global.cpu(), ood_segm)
                metric["meter"].add(val)

        # moose predictions
        for score_fn in ood_score_functions.keys():
            ood_scores_moose = ood_score_functions[score_fn]["ens"](all_preds)
            for key, metric in metrics[score_fn].items():
                val = metric["function"](ood_scores_moose.cpu(), ood_segm)
                metric["meter_moose"].add(val)

        t.update()
    t.close()

    print("\nAverage runtime = {:.05f}s".format(runtime_meter.mean))
    print("\nOoD detection results, {}:\n".format(args.dataset))
    print("{:>10} {:>10} {:>10}    {:>10}".format("", "", "AUPR", "FPR@95TPR"))
    for score_fn in ood_score_functions.keys():
        print("\n{:>10} {:>10} {:>10.02f}    {:>10.02f}".format(score_fn, "Global", metrics[score_fn]["AUPR"]["meter"].mean, metrics[score_fn]["FPR@95TPR"]["meter"].mean))
        print("{:>10} {:>10} {:>10.02f}    {:>10.02f}".format("", "MOoSe", metrics[score_fn]["AUPR"]["meter_moose"].mean, metrics[score_fn]["FPR@95TPR"]["meter_moose"].mean))
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--arch", type=str, default="deeplabv3_resnet50_multi_output")
    parser.add_argument("--probes-depth", type=int, default=1)
    parser.add_argument("--datasets-folder", default="./datasets/", type=str)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    pprint(args)
    main(args)

