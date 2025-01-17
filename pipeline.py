import argparse
import json
import os
import time

import cv2
import ffmpeg
import numpy as np
import torch
from PIL import Image
from supervision import VideoInfo

from summarization.DSNet.src.modules.model_zoo import get_model
from summarization.DSNet.src.kts.cpd_auto import cpd_auto
from yolov5.utils.general import check_img_size, xyxy2xywh
from action_detection.yowov2.config import build_dataset_config, build_model_config
from action_detection.yowov2.dataset.transforms import BaseTransform
from action_detection.yowov2.models import build_model
from action_detection.yowov2.utils.misc import load_weight
from fisheye.unfish import FisheyeFlatten
from summarization.DSNet.src.helpers import bbox_helper, vsumm_helper, init_helper
from summarization.deep_sort_pytorch.deep_sort import DeepSort
from summarization.deep_sort_pytorch.utils.parser import get_config
from summarization.DSNet.src.helpers.video_helper import FeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2 Demo')

    # basic
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='the size of input frame')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use cuda.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-vs', '--vis_thresh', default=0.3, type=float,
                        help='threshold for visualization')
    parser.add_argument('--video', default='9Y_l9NsnYE0.mp4', type=str,
                        help='AVA video name.')
    parser.add_argument('--gif', action='store_true', default=False,
                        help='generate gif.')

    # class label config
    parser.add_argument('-d', '--dataset', default='ava_v2.2',
                        help='ava_v2.2')
    parser.add_argument('--pose', action='store_true', default=False,
                        help='show 14 action pose of AVA.')

    # model
    parser.add_argument('-v', '--version', default='yowo_v2_large', type=str,
                        help='build YOWOv2')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='video clip length.')
    parser.add_argument('-m', '--memory', action="store_true", default=False,
                        help="memory propagate.")

    parser.add_argument('--deep_sort_weights', type=str,
                        default='summarization/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
                        help='ckpt.t7 path')
    parser.add_argument('--source', type=str, default='0', help='source')

    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')

    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')

    parser.add_argument("--config_deepsort", type=str, default="summarization/deep_sort_pytorch/configs/deep_sort.yaml")

    return parser.parse_args()


def kts(n_frames, features, sample_rate):
    seq_len = len(features)
    picks = np.arange(0, seq_len) * sample_rate

    # compute change points using KTS
    kernel = np.matmul(features, features.T)
    change_points, _ = cpd_auto(kernel, seq_len - 1, 1, verbose=False)
    change_points *= sample_rate
    change_points = np.hstack((0, change_points, n_frames))
    begin_frames = change_points[:-1]
    end_frames = change_points[1:]
    change_points = np.vstack((begin_frames, end_frames - 1)).T

    n_frame_per_seg = end_frames - begin_frames
    return change_points, n_frame_per_seg, picks


def multi_hot_vis(args, frame, out_bboxes, orig_w, orig_h, class_names, act_pose=False):
    action_list = [0] * 7

    for bbox in out_bboxes:
        x1, y1, x2, y2 = bbox[:4]

        cls_conf = bbox[5:]

        # rescale bbox
        x1, x2 = int(x1 * orig_w), int(x2 * orig_w)
        y1, y2 = int(y1 * orig_h), int(y2 * orig_h)

        det_conf = float(bbox[4])
        person_id = int(bbox[4])
        if det_conf < 0.7:
            continue
        cls_scores = np.sqrt(det_conf * cls_conf)

        indices = np.where(cls_scores > args.vis_thresh)
        scores = cls_scores[indices]
        indices = list(indices[0])
        scores = list(scores)

        if len(scores) > 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            blk = np.zeros(frame.shape, np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            coord = []
            text = []
            text_size = []

            for _, cls_ind in enumerate(indices):
                text.append("[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind]))
                # text.append("id " + str(person_id))
                text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.5, thickness=1)[0])
                coord.append((x1 + 3, y1 + 14))
                cv2.rectangle(blk, (coord[-1][0] - 1, coord[-1][1] - 12),
                              (coord[-1][0] + text_size[-1][0] + 1, coord[-1][1] + text_size[-1][1] - 4), (0, 255, 0),
                              cv2.FILLED)
                frame = cv2.addWeighted(frame, 1.0, blk, 0.5, 1)
                for t in range(len(text)):
                    cv2.putText(frame, text[t], coord[t], font, 0.5, (0, 0, 0), 1)

                action_list[cls_ind] += 1

    return frame, action_list


@torch.no_grad()
def detect(args, model, undistort, summarize, device, transform, class_names, class_colors):
    path_to_video = os.path.join(args.video)

    fps = 25.0
    save_size = (512, 512)
    save_name = os.path.join("/results", 'detection.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoCapture(str(path_to_video))
    out = cv2.VideoWriter(save_name, fourcc, fps, save_size)

    n_frames = 0

    video_clip = []
    frame_list = []
    feature_bank = []
    action_bank = {}

    sample_rate = 16

    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    googlenet_model = FeatureExtractor()

    while True:
        ret, frame = video.read()

        if ret:
            # flatten fisheye video
            flattened_frame = undistort(frame)

            frame_rgb = flattened_frame[..., (2, 1, 0)]
            frame_list.append(frame_rgb)

            frame_pil = Image.fromarray(frame_rgb.astype(np.uint8))

            video_clip.append(frame_pil)
            orig_h, orig_w = flattened_frame.shape[:2]
            n_frames += 1

            if len(video_clip) < sample_rate:
                continue

            # action detection
            x, _ = transform(video_clip)

            x = torch.stack(x, dim=1)
            x = x.unsqueeze(0).to(device)

            t0 = time.time()
            outputs = model(x)
            print("inference time ", time.time() - t0, "s")

            # get output
            batch_bboxes = outputs
            bboxes = batch_bboxes[0]

            xyxys = []
            confs = []
            clss = []
            act_label = []

            # transform to deepsort format input
            for det in bboxes:

                x1, y1, x2, y2 = det[0] * orig_w, det[1] * orig_h, det[2] * orig_w, det[3] * orig_w
                det_conf = float(det[4])

                if abs(x1 - x2) * abs(y2 - y1) < 600:
                    continue

                if x1 < 0 or x2 < 0 or y2 < 0 or y1 < 0:
                    continue

                if det_conf < 0.7:
                    continue

                xyxys.append([x1, y1, x2, y2])
                confs.append(det_conf)
                clss.append(0.)
                cls_conf = det[5:]

                indices = np.where(cls_conf > args.vis_thresh)
                indices = list(indices[0])
                if len(indices) != 0:
                    act_label.append(indices[0])
                else:
                    act_label.append(0)

            xyxys = torch.FloatTensor(xyxys)
            confs = torch.FloatTensor(confs)
            clss = torch.FloatTensor(clss)

            xywhs = xyxy2xywh(xyxys)

            ds_outputs = deepsort.update(xywhs, confs, clss, frame_rgb)

            if len(ds_outputs) != 0:
                action_bank[str(n_frames)] = []
                for i, output in enumerate(ds_outputs):
                    if i > len(clss) - 1:
                        break
                    actions = "{0}, [{1},{2},{3},{4}], {5}".format(output[4], output[0], output[1], output[2],
                                                                   output[3], int(act_label[i]))
                    action_bank[str(n_frames)].append(actions)

            # if len(bboxes) != 0:
            #     new_frame, action_list = multi_hot_vis(
            #         args=args,
            #         frame=flattened_frame,
            #         out_bboxes=bboxes,
            #         orig_w=orig_w,
            #         orig_h=orig_h,
            #         class_names=class_names,
            #         act_pose=args.pose
            #     )
            #
            #     frame_resized = cv2.resize(new_frame, (orig_h, orig_w))
            #
            #     out.write(frame_resized)
            #     cv2.imshow("frame", frame_resized)
            #     cv2.waitKey(1)
            #
            video_clip.clear()
        else:
            break
    video.release()
    out.release()

    # with open('action_bank2.csv', "w", newline='') as csvfile1:
    #     writer1 = csv.writer(csvfile1, delimiter=',')
    #     writer1.writerows(action_bank)

    with open("action_data.json", "w") as outfile:
        json.dump(action_bank, outfile)

    for i in range(len(frame_list)):
        if (i + 1) % sample_rate == 0:
            feature = googlenet_model.run(frame_list[i])
            feature_bank.append(feature)

    print("Start generate summary video")
    feature_bank = np.array(feature_bank)
    cps, nfps, picks = kts(n_frames, feature_bank, sample_rate)

    seq_len = len(feature_bank)

    with torch.no_grad():
        seq_torch = torch.from_numpy(feature_bank).unsqueeze(0).to(device)

        pred_cls, pred_bboxes = summarize.predict(seq_torch)

        pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)

        pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, args.nms_thresh)
        pred_summ = vsumm_helper.bbox2summary(
            seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)

    width = int(frame_list[0].shape[0])
    height = int(frame_list[0].shape[1])

    count_true = np.count_nonzero(pred_summ)
    save_path = None

    if count_true > 0:

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        save_path = f"F:/graduate_thesis/results/{args.save_path}"
        vid_writer = cv2.VideoWriter(save_path, fourcc, 25, (width, height))

        frame_idx = 0
        for i in range(len(frame_list)):

            if pred_summ[frame_idx]:
                frame = cv2.cvtColor(frame_list[i], cv2.COLOR_RGB2BGR)
                vid_writer.write(frame)
                cv2.imshow("frame", frame)
                cv2.waitkey(1)

            frame_idx += 1

        vid_writer.release()

        input_video = save_path
        output_video = save_path

        ffmpeg.input(input_video).output(output_video, vcodec='libx264').run(
            r"C:\Users\chang\Downloads\ffmpeg\bin\ffmpeg.exe", overwrite_output=True)

    else:
        print("No frames found")

    return action_bank, save_path, pred_summ


def pipeline(video):
    np.random.seed(100)
    args = parse_args()

    args.d = 'ava_v2.2'
    args.v = 'yowo_v2_large'
    args.size = 224
    args.weight = 'checkpoints/yowo2/yowo_v2_large.pth'
    args.video = video
    args.cuda = True
    args.save_path = 'short_video.mp4'

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    class_names = d_cfg['label_map']

    num_classes = 7

    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    video_path = os.path.join(os.getcwd(), args.video).replace("\\", "/")

    frames_size = VideoInfo.from_video_path(video_path).resolution_wh

    basetransform = BaseTransform(img_size=args.img_size)

    model, _ = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes,
        trainable=False
    )

    model = load_weight(model=model, path_to_ckpt=args.weight)

    if torch.cuda.is_available():
        model.cuda()

    summarize_args = init_helper.get_arguments()
    summarize_args.model = 'anchor-based'
    summarize_args.device = 'cuda'
    summarize_args.ckpt_path = 'checkpoints/dsnet/pretrain_ab_basic/checkpoint/summe.yml.4.pt'
    summarize_model = get_model(summarize_args.model, **vars(summarize_args))
    summarize_model = summarize_model.eval().to(summarize_args.device)
    state_dict = torch.load(summarize_args.ckpt_path, map_location=lambda storage, loc: storage)
    summarize_model.load_state_dict(state_dict)

    camera_matrix = np.load('checkpoints/fisheye/camera_matrix.npy')
    dist_coeffs = np.load('checkpoints/fisheye/dist_coeffs.npy')
    rectification = FisheyeFlatten(frames_size, 1, camera_matrix, dist_coeffs)

    print("Start detect")

    action_bank, save_path, pred_summ = detect(args=args,
                               model=model,
                               undistort=rectification,
                               summarize=summarize_model,
                               device=device,
                               transform=basetransform,
                               class_names=class_names,
                               class_colors=class_colors)

    return action_bank, save_path, pred_summ

if __name__ == '__main__':
    video = "D:/datasets/video3.mp4"
    pipeline(video)
