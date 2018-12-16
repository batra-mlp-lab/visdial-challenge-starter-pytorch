import argparse
import os

import cv2  # must import before importing caffe2 due to bug in cv2
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from tqdm import tqdm
import h5py
import numpy as np

from detectron.core.config import assert_and_infer_cfg, cfg, merge_cfg_from_file
from detectron.utils.boxes import nms as detectron_nms
from detectron.utils.io import cache_url
import detectron.utils.c2 as c2_utils
import detectron.core.test as detectron_test
import detectron.core.test_engine as infer_engine


c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

parser = argparse.ArgumentParser(description="Extract bottom-up features from a Detectron model")
parser.add_argument(
    "--config",
    help="Model config file used by Detectron",
    default="data/config_faster_rcnn_x101.yaml",
)
parser.add_argument(
    "--weights",
    help="Model weights file checkpointed from Detectron",
    default="data/model_faster_rcnn_x101.pkl",
)
parser.add_argument(
    "--output-h5",
    help="Path to output HDF file for saving bottom-up features",
    default="data/data_img_faster_rcnn_x101.h5",
)
parser.add_argument(
    "--max-boxes",
    help="Maximum number of bounding box proposals per image",
    type=int,
    default=36
)
parser.add_argument(
    "--image-root",
    help="Path to COCO train/val or VisDial val/test images",
)
parser.add_argument(
    "--split",
    help="Name of the split to process",
    choices=["train", "val", "test"]
)
parser.add_argument(
    "--feat-name",
    help="The name of the feature to extract, default: fc7",
    default="fc7"
)
parser.add_argument(
    "--gpu-id",
    help="The GPU id to use (-1 for CPU execution)",
    type=int,
    default=0
)


def get_detections_from_im(detectron_config,
                           detectron_model,
                           image,
                           feat_blob_name,
                           max_boxes=36,
                           conf_thresh=0.):
    assert conf_thresh >= 0

    with c2_utils.NamedCudaScope(0):
        scores, cls_boxes, im_scale = detectron_test.im_detect_bbox(
            detectron_model,
            image,
            detectron_config.TEST.SCALE,
            detectron_config.TEST.MAX_SIZE,
            boxes=None
        )
        num_proposals = scores.shape[0]

        rois = workspace.FetchBlob(f"gpu_{args.gpu_id}/rois")
        roi_features = workspace.FetchBlob(f"gpu_{args.gpu_id}/{feat_blob_name}")

        cls_boxes = rois[:, 1:5] / im_scale
        max_conf = np.zeros((num_proposals,), dtype=np.float32)
        max_cls = np.zeros((num_proposals,), dtype=np.int32)
        max_box = np.zeros((num_proposals, 4), dtype=np.float32)

        for cls_ind in range(1, detectron_config.MODEL.NUM_CLASSES):
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = np.array(detectron_nms(dets, detectron_config.TEST.NMS))
            idxs_update = np.where(cls_scores[keep] > max_conf[keep])
            keep_idxs = keep[idxs_update]
            max_conf[keep_idxs] = cls_scores[keep_idxs]
            max_cls[keep_idxs] = cls_ind
            max_box[keep_idxs] = dets[keep_idxs][:,:4]

        keep_boxes = np.where(max_conf > conf_thresh)[0]
        if len(keep_boxes) > max_boxes:
            keep_boxes = np.argsort(max_conf)[::-1][:max_boxes]

        objects = max_cls[keep_boxes]
        obj_confidence = max_conf[keep_boxes]
        obj_boxes = max_box[keep_boxes, :]
    return obj_boxes, roi_features[keep_boxes, :], objects, obj_confidence


def image_id_from_path(image_path):
    return int(image_path.split("/")[-1][-16:-4])


def main(args):
    # specifically for visual genome
    cfg.MODEL.NUM_ATTRIBUTES = -1

    merge_cfg_from_file(args.config)
    cfg.NUM_GPUS = 1
    cfg.TRAIN.CPP_RPN = 'none'
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    detectron_model = infer_engine.initialize_model_from_cfg(args.weights)

    image_paths = [os.path.join(args.image_root, name)
                   for name in os.listdir(args.image_root)]
    output_h5 = h5py.File(args.output_h5, "w")
    image_ids_h5d = output_h5.create_dataset(
        "image_ids", (len(image_paths), )
    )

    # TODO: remove hardcoded 2048
    features_h5d = output_h5.create_dataset(
        "features", (len(image_paths), args.max_boxes, 2048),
        chunks=(1, args.max_boxes, 2048)
    )

    for idx, image_path in tqdm(enumerate(image_paths)):
        image_ids_h5d[idx] = image_id_from_path(image_path)

        image = cv2.imread(image_path)
        _, bottomup_features, _, _ = get_detections_from_im(
            cfg,
            detectron_model,
            image,
            args.feat_name,
            args.max_boxes)
        features_h5d[idx] = bottomup_features
    output_h5.close()


if __name__ == "__main__":
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    args = parser.parse_args()
    main(args)
