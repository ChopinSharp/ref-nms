import os
import random
import pickle
import math

import torch
from torch.utils.data import Dataset
from torchvision.ops import nms
import numpy as np
import h5py
from tqdm import trange

from utils.misc import mrcn_crop_pool_layer, recursive_jitter_roi, repeat_loader, calculate_iou


__all__ = ['RankDataset', 'RankEvalLoader', 'RankEvaluator']


class RankDataset(Dataset):

    # Pre-extracted image feature files: {image_id}.h5
    # Format: {'head': (1, 1024, ih, iw), 'im_info': [[ih, iw, scale]]}
    # ih == im_height*scale/16.0, iw == im_width*scale/16.0)
    HEAD_FEAT_DIR = 'data/head_feats'
    BOX_FILE_PATH = 'data/rpn_boxes.pkl'
    SCORE_FILE_PATH = 'data/rpn_box_scores.pkl'
    CONF_THRESH = 0.05

    def __init__(self, refdb, ctxdb, split, level_num, roi_per_level, negative_num):
        Dataset.__init__(self)
        self.refs = refdb[split]
        self.dataset_splitby = refdb['dataset_splitby']
        self.exp_to_ctx = ctxdb[split]
        self.idx_to_glove = np.load('cache/std_glove_{}.npy'.format(refdb['dataset_splitby']))
        self.max_sent_len = 20 if refdb['dataset_splitby'] == 'refcocog_umd' else 10
        self.pad_feat = np.zeros(300, dtype=np.float32)
        self.level_num = level_num
        end_points = np.linspace(0.5, 1.0, num=level_num, endpoint=True).tolist()
        self.interval_list = list(zip(end_points[:-1], end_points[1:]))
        self.interval_list.insert(0, (0.1, 0.4))
        self.roi_num_list = [negative_num] + (level_num - 1) * [roi_per_level]
        with open(self.BOX_FILE_PATH, 'rb') as f:
            self.img_to_det_box = pickle.load(f)
        with open(self.SCORE_FILE_PATH, 'rb') as f:
            self.img_to_det_score = pickle.load(f)

    def __getitem__(self, idx):
        """

        Returns:
            roi_feat: [R, 1024, 7, 7]
            word_feat: [S, 300]

        """
        # Index refdb entry
        ref = self.refs[idx]
        image_id = ref['image_id']
        gt_box = ref['bbox']
        exp_id = ref['exp_id']
        ctx_list = self.exp_to_ctx[str(exp_id)]['ctx']
        # Build word features
        word_feat, sent_len = self.build_word_feats(ref['tokens'])
        # Load image feature
        image_h5 = h5py.File(os.path.join(self.HEAD_FEAT_DIR, self.dataset_splitby, '{}.h5'.format(image_id)), 'r')
        scaled_h, scaled_w, scale = image_h5['im_info'][0].tolist()
        image_feat = torch.tensor(image_h5['head'])  # [1, 1024, ih, iw]
        # Jitter ROIs
        roi_list = self.get_roi_list(image_id, gt_box, ctx_list, scale, scaled_w, scaled_h)
        roi_t = torch.tensor(roi_list)
        roi_feat = mrcn_crop_pool_layer(image_feat, roi_t)
        return roi_feat, word_feat, sent_len

    def __len__(self):
        return len(self.refs)

    def get_roi_list(self, image_id, gt_box, ctx_list, scale, scaled_w, scaled_h):
        # Bin detection boxes according to IoU and scale them
        boxes = self.img_to_det_box[image_id].reshape(-1, 81, 4)
        scores = self.img_to_det_score[image_id]
        boxes = boxes[:, 1:]    # [*, 80, 4]
        scores = scores[:, 1:]  # [*, 80]
        box_list = [[] for _ in range(self.level_num)]
        target_list = [gt_box]
        for t in ctx_list:
            target_list.append(t['box'])
        for box in boxes[scores > self.CONF_THRESH]:
            iou = max([calculate_iou(t, box) for t in target_list])
            level = math.ceil(max(0, (iou - 0.5)) * (self.level_num - 1) / 0.5)
            box_list[level].append(self.scale_roi(box, scale, scaled_w, scaled_h))
        # Construct RoI list
        scaled_target_list = [self.scale_roi(t, scale, scaled_w, scaled_h) for t in target_list]
        roi_list = []
        for (L, R), level_roi_num, level_box_list in zip(self.interval_list, self.roi_num_list, box_list):
            sampled_boxes, less_num = self.sample_roi(level_box_list, level_roi_num)
            roi_list.extend(sampled_boxes)
            for _ in range(less_num):
                scaled_t = random.choice(scaled_target_list)
                roi_list.append(recursive_jitter_roi(scaled_t, L, R, scaled_w, scaled_h))
        assert len(roi_list) == sum(self.roi_num_list)
        return roi_list

    @staticmethod
    def scale_roi(roi, scale, scaled_w, scaled_h):
        x0, y0, x1, y1 = roi
        scaled_x0 = min(x0 * scale, scaled_w)
        scaled_y0 = min(y0 * scale, scaled_h)
        scaled_x1 = min(x1 * scale, scaled_w)
        scaled_y1 = min(y1 * scale, scaled_h)
        return scaled_x0, scaled_y0, scaled_x1, scaled_y1

    @staticmethod
    def sample_roi(candidate_list, num):
        candidate_num = len(candidate_list)
        if candidate_num == 0:
            return [], num
        elif candidate_num <= num:
            return candidate_list, num - candidate_num
        else:
            return random.sample(candidate_list, num), 0

    def build_word_feats(self, tokens):
        word_feats = [self.idx_to_glove[wd_idx] for wd_idx in tokens]
        word_feats += [self.pad_feat] * max(self.max_sent_len - len(word_feats), 0)
        word_feats = torch.tensor(word_feats[:self.max_sent_len])  # [S, 300]
        return word_feats, min(len(tokens), self.max_sent_len)


class RankEvalLoader:

    BOX_FILE_PATH = 'cache/rpn_boxes.pkl'
    SCORE_FILE_PATH = 'cache/rpn_box_scores.pkl'
    IMG_FEAT_DIR = 'cache/head_feats/matt-mrcn'

    def __init__(self, refdb, split='val', conf_thresh=.05):
        self.refs = refdb[split]
        self.img_to_exps = {}
        for ref in self.refs:
            image_id = ref['image_id']
            if image_id in self.img_to_exps:
                self.img_to_exps[image_id].append((ref['exp_id'], ref['tokens']))
            else:
                self.img_to_exps[image_id] = [(ref['exp_id'], ref['tokens'])]
        with open(self.BOX_FILE_PATH, 'rb') as f:
            self.img_to_det_box = pickle.load(f)
        with open(self.SCORE_FILE_PATH, 'rb') as f:
            self.img_to_det_score = pickle.load(f)
        self.idx_to_glove = np.load('cache/glove_{}.npy'.format(refdb['dataset_splitby']))
        self.conf_thresh = conf_thresh

    def __iter__(self):
        for image_id, exps in self.img_to_exps.items():
            # Load image feature
            image_h5 = h5py.File(os.path.join(self.IMG_FEAT_DIR, '{}.h5'.format(image_id)), 'r')
            scale = image_h5['im_info'][0, 2]
            image_feat = torch.tensor(image_h5['head'])  # [1, 1024, ih, iw]
            # RoI-pool positive M-RCNN detections
            det_box = self.img_to_det_box[image_id].reshape(-1, 81, 4)  # [300, 81, 4]
            det_score = self.img_to_det_score[image_id]  # [300, 81]
            det_box = np.transpose(det_box[:, 1:], axes=[1, 0, 2])  # [80, 300, 4]
            det_score = np.transpose(det_score[:, 1:], axes=[1, 0])  # [80, 300]
            positive = det_score > self.conf_thresh    # [80, 300]
            pos_box = torch.tensor(det_box[positive])  # [*, 4]
            pos_score = torch.tensor(det_score[positive])  # [*]
            cls_num_list = np.sum(positive, axis=1).tolist()  # [80]
            pos_feat = mrcn_crop_pool_layer(image_feat, pos_box * scale)  # [*, 1024, 7, 7]
            pos_feat = pos_feat.unsqueeze(0)  # [1, *, 1024, 7, 7]
            for exp_id, tokens in exps:
                # Load word feature
                assert isinstance(tokens, list)
                sent_feat = torch.tensor(self.idx_to_glove[tokens])
                sent_feat = sent_feat.unsqueeze(0)  # [1, *, 300]
                yield exp_id, pos_feat, sent_feat, pos_box, pos_score, cls_num_list

    def __len__(self):
        return len(self.refs)


class RankEvaluator:

    def __init__(self, refdb, split, num_sample, top_N=None, gpu_id=0, alpha=0.15):
        """Runtime ref-based hit rate evaluator.

        Args:
            refdb: `refdb` dict.
            split: Dataset split to evaluate on.
            top_N: Select top-N scoring proposals to evaluate. `None` means no selection. Default `None`.
            num_sample: Use `num_sample` refs to evaluate hit rate.

        """
        self.dataset_splitby = refdb['dataset_splitby']
        self.exp_to_box = {}
        for ref in refdb[split]:
            self.exp_to_box[ref['exp_id']] = ref['bbox']
        self.split = split
        self.top_N = top_N
        loader = RankEvalLoader(refdb, split=split, conf_thresh=0.05)
        self.loader = repeat_loader(loader)
        self.total = len(loader)
        self.num_sample = num_sample
        self.device = torch.device('cuda', gpu_id)
        self.alpha = alpha

    def eval_hit_rate(self, predictor):
        """Estimate hit rate with `num_sample` samples during runtime.

        Args:
            predictor: `torch.nn.module` to evaluate. Module should be set to eval mode IN ADVANCE.
                All parameters of predictor has to be on the SAME device.

        Returns:
            proposal_per_ref: Average proposal number per referring expression.
            hit_rate: Estimated hit rate.

        """
        print('{} expressions in {} {} split, using {} samples to evaluate hit rate...'
              .format(self.total, self.dataset_splitby, self.split, self.num_sample))
        # Use predictor to score proposals
        exp_to_proposals = {}
        for _ in trange(self.num_sample, desc='Estimating hit rate', ascii=True):
            exp_id, pos_feat, sent_feat, pos_box, pos_score, cls_num_list = next(self.loader)
            pos_feat = pos_feat.to(self.device)    # [1, R, 1024, 7, 7]
            sent_feat = sent_feat.to(self.device)  # [1, S, 300]
            pos_box = pos_box.to(self.device)
            pos_score = pos_score.to(self.device)
            with torch.no_grad():
                rank_score, *_ = predictor(pos_feat, sent_feat)  # [1, R]
            rank_score = rank_score.squeeze(dim=0)
            rank_score_list = torch.split(rank_score, cls_num_list, dim=0)
            pos_box_list = torch.split(pos_box, cls_num_list, dim=0)
            pos_score_list = torch.split(pos_score, cls_num_list, dim=0)
            proposals = []
            for cls_rank_score, cls_pos_box, cls_pos_score in zip(rank_score_list, pos_box_list, pos_score_list):
                # No positive box under this category
                if cls_rank_score.size(0) == 0:
                    continue
                final_score = self.alpha * cls_rank_score + (1 - self.alpha) * cls_pos_score
                keep = nms(cls_pos_box, final_score, iou_threshold=0.3)
                cls_kept_box = cls_pos_box[keep]
                cls_kept_score = final_score[keep]
                for box, score in zip(cls_kept_box, cls_kept_score):
                    proposals.append({'score': score.item(), 'box': box.tolist()})
            exp_to_proposals[exp_id] = proposals
        # Estimate hit rate
        assert len(exp_to_proposals) == self.num_sample
        num_proposal = 0
        num_hit = 0
        for exp_id, proposals in exp_to_proposals.items():
            ranked_proposals = sorted(proposals, key=lambda p: p['score'], reverse=True)[:self.top_N]
            gt_box = self.exp_to_box[exp_id]
            num_proposal += len(ranked_proposals)
            for proposal in ranked_proposals:
                if calculate_iou(gt_box, proposal['box']) > 0.5:
                    num_hit += 1
                    break
        proposal_per_ref = num_proposal / self.num_sample
        hit_rate = num_hit / self.num_sample
        return proposal_per_ref, hit_rate
