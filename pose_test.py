from typing import List
import sys
import cv2
import torch
from torch.nn import functional as F

import numpy as np
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

import pose
# from densepose.config import add_densepose_config, add_hrnet_config
# from densepose.vis.densepose import (
#     DensePoseResultsContourVisualizer,
#     DensePoseResultsFineSegmentationVisualizer,
#     DensePoseResultsUVisualizer,
#     DensePoseResultsVVisualizer,
# )
# from densepose.vis.extractor import DensePoseResultExtractor, extract_boxes_xywh_from_instances

import time


# def setup_densepose(config_fpath: str, model_fpath: str, opts: List[str]):
#     cfg = get_cfg()
#     add_densepose_config(cfg)
#     add_hrnet_config(cfg)
#     cfg.merge_from_file(config_fpath)
#     # cfg.merge_from_list(args.opts)
#     if opts:
#         cfg.merge_from_list(opts)
#     cfg.MODEL.WEIGHTS = model_fpath
#     cfg.freeze()
#     return cfg


def setup_keypoints(conff):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(conff))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(conff)

    cfg.freeze()
    return cfg


# class DensePoseResultCustomExtractor(object):
#     """
#     Extracts DensePose result from instances
#     """

#     def __call__(self, instances, select=None):
#         boxes_xywh = extract_boxes_xywh_from_instances(instances)
#         if instances.has("pred_densepose") and (boxes_xywh is not None):
#             dpout = instances.pred_densepose
#             if select is not None:
#                 dpout = dpout[select]
#                 boxes_xywh = boxes_xywh[select]
#             return convert_results(boxes_xywh, dpout.S, dpout.I, dpout.U, dpout.V)
#         else:
#             return None


# def convert_results(boxes_xywh, S, I, U, V):
#     return [convert_result(box_xywh, S[[i]], I[[i]], U[[i]], V[[i]]) for i, box_xywh in enumerate(boxes_xywh)]


# def convert_result(box_xywh, S, I, U, V):
#     # TODO: reuse resample_output_to_bbox
#     x, y, w, h = box_xywh
#     w = max(int(w), 1)
#     h = max(int(h), 1)
#     result = torch.zeros([3, h, w], dtype=torch.uint8, device=U.device)
#     assert (
#         len(S.size()) == 4
#     ), "AnnIndex tensor size should have {} " "dimensions but has {}".format(4, len(S.size()))
#     s_bbox = F.interpolate(S, (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
#     assert (
#         len(I.size()) == 4
#     ), "IndexUV tensor size should have {} " "dimensions but has {}".format(4, len(S.size()))
#     i_bbox = (
#         F.interpolate(I, (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
#         * (s_bbox > 0).long()
#     ).squeeze(0)

#     # np.set_printoptions(threshold=sys.maxsize)
#     # print(box_xywh)
#     # print(i_bbox.cpu().tolist())
#     return i_bbox


# def check_body_part(input_tensor):
#     # coarse segmentation: 1 = Torso, 2 = Right Hand, 3 = Left Hand,
#     # 4 = Left Foot, 5 = Right Foot, 6 = Upper Leg Right, 7 = Upper Leg Left,
#     # 8 = Lower Leg Right, 9 = Lower Leg Left, 10 = Upper Arm Left,
#     # 11 = Upper Arm Right, 12 = Lower Arm Left, 13 = Lower Arm Right,
#     # 14 = Head
#     # fine segmentation: 1, 2 = Torso, 3 = Right Hand, 4 = Left Hand,
#     # 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right,
#     # 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right,
#     # 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left,
#     # 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left,
#     # 20, 22 = Lower Arm Right, 23, 24 = Head

#     def check_exist(label):
#         return (input_tensor == label).nonzero().size()[0] != 0

#     results = []
#     if check_exist(23) or check_exist(24):
#         results.append('Head')
#     if check_exist(1) or check_exist(2):
#         results.append('Torso')
#     if check_exist(16) or check_exist(18):
#         results.append('RUArm')
#     if check_exist(20) or check_exist(22):
#         results.append('RLArm')
#     if check_exist(15) or check_exist(17):
#         results.append('LUArm')
#     if check_exist(19) or check_exist(21):
#         results.append('LLArm')
#     if check_exist(3) :
#         results.append('RHand')
#     if check_exist(4) or check_exist(24):
#         results.append('LHand')
#     if check_exist(7) or check_exist(9):
#         results.append('RULeg')
#     if check_exist(11) or check_exist(13):
#         results.append('RLLeg')
#     if check_exist(8) or check_exist(10):
#         results.append('LULeg')
#     if check_exist(12) or check_exist(14):
#         results.append('LLLeg')
#     if check_exist(6):
#         results.append('RFoot')
#     if check_exist(5):
#         results.append('LFoot')

#     # print((input_tensor == 1).nonzero())
#     # print((input_tensor == 23).nonzero())
#     return results


def main():
    conf_threshold = 0.7

    conff = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"

    # load config from file and command-line arguments

    # keypoint.
    cfg = setup_keypoints(conff)

    # Densepose.
    # config_file = '/home/bogonets/Projects/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml'
    # model_file = '/home/bogonets/Projects/detectron2/projects/DensePose/model_final_162be9.pkl'
    # cfg = setup_densepose(config_file, model_file, [])
    # extractor = DensePoseResultCustomExtractor()
    # visualizer = DensePoseResultsContourVisualizer()
    

    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )

    cpu_device = torch.device("cpu")

    predictor = DefaultPredictor(cfg)

    # cap = cv2.VideoCapture("/media/bogonets/Data/Video/nambu/5.avi")
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow("pose", cv2.WINDOW_KEEPRATIO)

    times = []
    while True:
        ret, frame = cap.read()

        if not ret:
            print("cap is wrong.")
            break

        now = time.time()
        times = [ t for t in times if now - t <= 1]
        times.append(now)
        print(f"fps : {len(times)}")

        # DensePose.
        # with torch.no_grad():
        #     outputs = predictor(frame)["instances"]
        #     out = frame
        #     data = extractor(outputs)

        #     for idx, d in enumerate(data):
        #         if d.nonzero().size()[0] == 0:
        #             continue
        #         parts = check_body_part(d)
        #         print(parts)
        #         labeled_img = 255 - (d.cpu().numpy() * 30)
        #         labeled_img = labeled_img.astype('uint8')
        #         cv2.imshow(f"idx {idx}", labeled_img)
        #         if 'LHand' in parts:
        #             temp_d = d.clone()
        #             temp_d[temp_d != 3] = 255
        #             temp_d[temp_d == 3] = 0
        #             labeled_img = temp_d.cpu().numpy()
        #             labeled_img = labeled_img.astype('uint8')
        #             cv2.imshow(f"idx {idx} - r hand", labeled_img)
        #         if 'LHand' in parts:
        #             temp_d = d.clone()
        #             temp_d[temp_d != 4] = 255
        #             temp_d[temp_d == 4] = 0
        #             labeled_img = temp_d.cpu().numpy()
        #             labeled_img = labeled_img.astype('uint8')
        #             cv2.imshow(f"idx {idx} - l hand", labeled_img)
        #     # image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     # image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        #     # out = visualizer.visualize(image, data)

        # Keypoints.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = predictor(frame)

        instances = predictions["instances"].to(cpu_device)

        keypoints = instances.pred_keypoints if instances.has(
            "pred_keypoints") else None

        # print(keypoints)
        # print(keypoints.numpy())

        persons = pose.xposeOverShoulderByPrediction(keypoints, 0.4)
        for result in persons:
            if 'result' in result and result['result']:
                print("!!!XXXXXX!!!!")
                frame = cv2.putText(frame, "X", (100, 100),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 5)

            if 'left' in result:
                p1, p2 = result['left']
                cv2.line(frame, (int(p1[0]), int(p1[1])),
                         (int(p2[0]), int(p2[1])), (0, 0, 255), 20)

            if 'right' in result:
                p1, p2 = result['right']
                cv2.line(frame, (int(p1[0]), int(p1[1])),
                         (int(p2[0]), int(p2[1])), (0, 0, 255), 20)

            if 'extend_left' in result:
                p1, p2 = result['extend_left']
                cv2.line(frame, (int(p1[0]), int(p1[1])),
                         (int(p2[0]), int(p2[1])), (255, 0, 0), 20)

            if 'extend_right' in result:
                p1, p2 = result['extend_right']
                cv2.line(frame, (int(p1[0]), int(p1[1])),
                         (int(p2[0]), int(p2[1])), (255, 0, 0), 20)

        vis = Visualizer(frame, metadata, instance_mode=ColorMode.SEGMENTATION)
        out = vis.draw_instance_predictions(instances)
        out = out.get_image()[:, :, ::-1]

        # Show Image.
        # cv2.imshow("ori", frame)
        cv2.imshow("pose", out)
        key = cv2.waitKey(1)

        if key == 27:
            break

    cap.release()


if __name__ == "__main__":
    main()
