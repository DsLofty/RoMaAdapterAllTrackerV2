"""Dataset helpers shared by V2 final-stage export/evaluation scripts."""

from __future__ import annotations

import os


KNOWN_TAPVID_SUBDIRS = {
    "dav": "tapvid_davis",
    "kin": "tapvid_kinetics",
    "rgb": "tapvid_rgb_stacking",
    "rob": "robotap",
}


def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))


def resolve_dataset_root(args):
    if args.data_dir:
        path = expand_path(args.data_dir)
        base = os.path.basename(os.path.normpath(path))
        if base in set(KNOWN_TAPVID_SUBDIRS.values()):
            return os.path.dirname(path)
        return path
    return os.path.join(os.path.expanduser("~/Datasets"), "tap_vid")


def get_dataset(args):
    dataset_root = resolve_dataset_root(args)
    if args.dname == "dav":
        from datasets import davisdataset

        dataset = davisdataset.DavisDataset(
            data_root=os.path.join(dataset_root, "tapvid_davis"),
            crop_size=args.image_size,
            only_first=args.only_first,
        )
    elif args.dname == "kin":
        from datasets import kineticsdataset

        dataset = kineticsdataset.KineticsDataset(
            data_root=os.path.join(dataset_root, "tapvid_kinetics"),
            crop_size=args.image_size,
            only_first=True,
        )
    elif args.dname == "rgb":
        from datasets import rgbstackingdataset

        dataset = rgbstackingdataset.RGBStackingDataset(
            data_root=os.path.join(dataset_root, "tapvid_rgb_stacking"),
            crop_size=args.image_size,
            only_first=args.only_first,
        )
    elif args.dname == "rob":
        from datasets import robotapdataset

        dataset = robotapdataset.RobotapDataset(
            data_root=os.path.join(dataset_root, "robotap"),
            crop_size=args.image_size,
            only_first=True,
        )
    else:
        raise ValueError("unsupported TAPVID dname: %s" % args.dname)
    return dataset, dataset_root


def safe_seq_id(batch, fallback_index):
    seq_name = getattr(batch, "seq_name", None)
    if isinstance(seq_name, list) and seq_name and seq_name[0] is not None:
        return str(seq_name[0])
    return "seq_%03d" % int(fallback_index)
