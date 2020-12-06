import pickle
import os
import random

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py

from utils.misc import calculate_iou, mrcn_crop_pool_layer


__all__ = ['DetBoxDataset', 'DetEvalLoader', 'DetEvalTopLoader']


class DetBoxDataset(Dataset):

    HEAD_FEAT_DIR = 'data/head_feats'
    BOX_FILE_PATH = 'data/rpn_boxes.pkl'
    SCORE_FILE_PATH = 'data/rpn_box_scores.pkl'
    CONF_THRESH = 0.05
    DELTA_CONF = 0.005

    def __init__(self, refdb, ctxdb, split, roi_per_img):
        Dataset.__init__(self)
        self.refs = refdb[split]
        self.dataset_splitby = refdb['dataset_splitby']
        self.exp_to_ctx = ctxdb[split]
        with open(self.BOX_FILE_PATH, 'rb') as f:
            self.img_to_det_box = pickle.load(f)
        with open(self.SCORE_FILE_PATH, 'rb') as f:
            self.img_to_det_score = pickle.load(f)
        self.idx_to_glove = np.load('cache/std_glove_{}.npy'.format(refdb['dataset_splitby']))
        self.max_sent_len = 20 if refdb['dataset_splitby'] == 'refcocog_umd' else 10
        self.pad_feat = np.zeros(300, dtype=np.float32)
        # Number of samples to draw from one image
        self.roi_per_img = roi_per_img

    def __getitem__(self, idx):
        """

        Returns:
            roi_feats: [R, 1024, 7, 7]
            roi_labels: [R]
            word_feats: [S, 300]
            sent_len: [0]

        """
        # Index refer object
        ref = self.refs[idx]
        image_id = ref['image_id']
        gt_box = ref['bbox']
        exp_id = ref['exp_id']
        ctx_boxes = [c['box'] for c in self.exp_to_ctx[str(exp_id)]['ctx']]
        target_list = [gt_box] + ctx_boxes
        pos_rois, neg_rois = self.get_labeled_rois(image_id, target_list)
        # Build word features
        word_feats, sent_len = self.build_word_feats(ref['tokens'])
        # Load image feature
        image_h5 = h5py.File(os.path.join(self.HEAD_FEAT_DIR, self.dataset_splitby, '{}.h5'.format(image_id)), 'r')
        scale = image_h5['im_info'][0, 2]
        image_feat = torch.tensor(image_h5['head'])  # [1, 1024, ih, iw]
        # Sample ROIs
        pos_num = min(len(pos_rois), self.roi_per_img // 2)
        neg_num = min(len(neg_rois), self.roi_per_img - pos_num)
        pos_num = self.roi_per_img - neg_num
        sampled_pos = random.sample(pos_rois, pos_num)
        sampled_neg = random.sample(neg_rois, neg_num)
        pos_labels = torch.ones(len(sampled_pos), dtype=torch.float)
        neg_labels = torch.zeros(len(sampled_neg), dtype=torch.float)
        roi_labels = torch.cat([pos_labels, neg_labels], dim=0)  # [R]
        # Extract head features
        sampled_roi = torch.tensor(sampled_pos + sampled_neg)    # [R, 4]
        sampled_roi.mul_(scale)
        roi_feats = mrcn_crop_pool_layer(image_feat, sampled_roi)
        return roi_feats, roi_labels, word_feats, sent_len

    def __len__(self):
        return len(self.refs)

    def get_labeled_rois(self, image_id, target_list):
        boxes = self.img_to_det_box[image_id].reshape(-1, 81, 4)
        scores = self.img_to_det_score[image_id]
        boxes = boxes[:, 1:]  # [*, 80, 4]
        scores = scores[:, 1:]  # [*, 80]
        # boxes = boxes.reshape(-1, 4)
        # scores = scores.reshape(-1)
        # top_idx = np.argsort(scores)[-self.TOP_N:]
        this_thresh = self.CONF_THRESH
        positive = scores > this_thresh
        while np.sum(positive) < self.roi_per_img:
            this_thresh -= self.DELTA_CONF
            positive = scores > this_thresh
        pos_rois = []
        neg_rois = []
        # for box in boxes[top_idx]:
        for box in boxes[positive]:
            for t in target_list:
                if calculate_iou(box, t) >= 0.5:
                    pos_rois.append(box)
                    break
            else:
                neg_rois.append(box)
        return pos_rois, neg_rois

    def build_word_feats(self, tokens):
        word_feats = [self.idx_to_glove[wd_idx] for wd_idx in tokens]
        word_feats += [self.pad_feat] * max(self.max_sent_len - len(word_feats), 0)
        word_feats = torch.tensor(word_feats[:self.max_sent_len])  # [S, 300]
        return word_feats, min(len(tokens), self.max_sent_len)


class DetBoxDatasetNoCtx(Dataset):

    HEAD_FEAT_DIR = 'cache/head_feats/matt-mrcn'
    BOX_FILE_PATH = 'cache/rpn_boxes.pkl'
    SCORE_FILE_PATH = 'cache/rpn_box_scores.pkl'
    CONF_THRESH = 0.05
    DELTA_CONF = 0.005

    def __init__(self, refdb, split, roi_per_img):
        Dataset.__init__(self)
        self.refs = refdb[split]
        with open(self.BOX_FILE_PATH, 'rb') as f:
            self.img_to_det_box = pickle.load(f)
        with open(self.SCORE_FILE_PATH, 'rb') as f:
            self.img_to_det_score = pickle.load(f)
        self.idx_to_glove = np.load('cache/std_glove_{}.npy'.format(refdb['dataset_splitby']))
        self.max_sent_len = 20 if refdb['dataset_splitby'] == 'refcocog_umd' else 10
        self.pad_feat = np.zeros(300, dtype=np.float32)
        # Number of samples to draw from one image
        self.roi_per_img = roi_per_img

    def __getitem__(self, idx):
        """

        Returns:
            roi_feats: [R, 1024, 7, 7]
            roi_labels: [R]
            word_feats: [S, 300]
            sent_len: [0]

        """
        # Index refer object
        ref = self.refs[idx]
        image_id = ref['image_id']
        gt_box = ref['bbox']
        pos_rois, neg_rois = self.get_labeled_rois(image_id, gt_box)
        # Build word features
        word_feats, sent_len = self.build_word_feats(ref['tokens'])
        # Load image feature
        image_h5 = h5py.File(os.path.join(self.HEAD_FEAT_DIR, '{}.h5'.format(image_id)), 'r')
        scale = image_h5['im_info'][0, 2]
        image_feat = torch.tensor(image_h5['head'])  # [1, 1024, ih, iw]
        # Sample ROIs
        pos_num = min(len(pos_rois), self.roi_per_img // 2)
        neg_num = min(len(neg_rois), self.roi_per_img - pos_num)
        pos_num = self.roi_per_img - neg_num
        sampled_pos = random.sample(pos_rois, pos_num)
        sampled_neg = random.sample(neg_rois, neg_num)
        pos_labels = torch.ones(len(sampled_pos), dtype=torch.float)
        neg_labels = torch.zeros(len(sampled_neg), dtype=torch.float)
        roi_labels = torch.cat([pos_labels, neg_labels], dim=0)  # [R]
        # Extract head features
        sampled_roi = torch.tensor(sampled_pos + sampled_neg)    # [R, 4]
        sampled_roi.mul_(scale)
        roi_feats = mrcn_crop_pool_layer(image_feat, sampled_roi)
        return roi_feats, roi_labels, word_feats, sent_len

    def __len__(self):
        return len(self.refs)

    def get_labeled_rois(self, image_id, gt_box):
        boxes = self.img_to_det_box[image_id].reshape(-1, 81, 4)
        scores = self.img_to_det_score[image_id]
        boxes = boxes[:, 1:]  # [*, 80, 4]
        scores = scores[:, 1:]  # [*, 80]
        this_thresh = self.CONF_THRESH
        positive = scores > this_thresh
        while np.sum(positive) < self.roi_per_img:
            this_thresh -= self.DELTA_CONF
            positive = scores > this_thresh
        pos_rois = []
        neg_rois = []
        # for box in boxes[top_idx]:
        for box in boxes[positive]:
            if calculate_iou(box, gt_box) >= 0.5:
                pos_rois.append(box)
            else:
                neg_rois.append(box)
        return pos_rois, neg_rois

    def build_word_feats(self, tokens):
        word_feats = [self.idx_to_glove[wd_idx] for wd_idx in tokens]
        word_feats += [self.pad_feat] * max(self.max_sent_len - len(word_feats), 0)
        word_feats = torch.tensor(word_feats[:self.max_sent_len])  # [S, 300]
        return word_feats, min(len(tokens), self.max_sent_len)


class DetEvalLoader:

    BOX_FILE_PATH = 'data/rpn_boxes.pkl'
    SCORE_FILE_PATH = 'data/rpn_box_scores.pkl'
    IMG_FEAT_DIR = 'data/head_feats'
    CONF_THRESH = 0.05
    DELTA_CONF = 0.005

    def __init__(self, refdb, split='val', gpu_id=0):
        self.dataset_splitby = refdb['dataset_splitby']
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
        self.idx_to_glove = np.load('cache/std_glove_{}.npy'.format(refdb['dataset_splitby']))
        self.device = torch.device('cuda', gpu_id)

    def __iter__(self):
        # Fetch ref info
        for image_id, exps in self.img_to_exps.items():
            # Load image feature
            image_h5 = h5py.File(os.path.join(self.IMG_FEAT_DIR, self.dataset_splitby, '{}.h5'.format(image_id)), 'r')
            scale = image_h5['im_info'][0, 2]
            image_feat = torch.tensor(image_h5['head'])  # [1, 1024, ih, iw]
            # RoI-pool positive M-RCNN detections
            det_box = self.img_to_det_box[image_id].reshape(-1, 81, 4)  # [300, 81, 4]
            det_score = self.img_to_det_score[image_id]  # [300, 81]
            det_box = np.transpose(det_box[:, 1:], axes=[1, 0, 2])  # [80, 300, 4]
            det_score = np.transpose(det_score[:, 1:], axes=[1, 0])  # [80, 300]
            this_thresh = self.CONF_THRESH
            positive = det_score > this_thresh  # [80, 300]
            while np.sum(positive) == 0:
                this_thresh -= self.DELTA_CONF
                positive = det_score > this_thresh  # [80, 300]
            pos_box = torch.tensor(det_box[positive])  # [*, 4]
            pos_score = torch.tensor(det_score[positive], device=self.device)  # [*]
            cls_num_list = np.sum(positive, axis=1).tolist()  # [80]
            pos_feat = mrcn_crop_pool_layer(image_feat, pos_box * scale)  # [*, 1024, 7, 7]
            pos_feat = pos_feat.to(self.device).unsqueeze(0)  # [1, *, 1024, 7, 7]
            pos_box = pos_box.to(self.device)
            for exp_id, tokens in exps:
                # Load word feature
                assert isinstance(tokens, list)
                sent_feat = torch.tensor(self.idx_to_glove[tokens], device=self.device)
                sent_feat = sent_feat.unsqueeze(0)  # [1, *, 300]
                yield exp_id, pos_feat, sent_feat, pos_box, pos_score, cls_num_list

    def __len__(self):
        return len(self.refs)


class DetEvalTopLoader:

    BOX_FILE_PATH = 'cache/rpn_boxes.pkl'
    SCORE_FILE_PATH = 'cache/rpn_box_scores.pkl'
    IMG_FEAT_DIR = 'cache/head_feats/matt-mrcn'

    def __init__(self, refdb, split='val', gpu_id=0, top_N=200):
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
        self.idx_to_glove = np.load('cache/std_glove_{}.npy'.format(refdb['dataset_splitby']))
        self.device = torch.device('cuda', gpu_id)
        self.top_N = top_N

    def __iter__(self):
        # Fetch ref info
        for image_id, exps in self.img_to_exps.items():
            # Load image feature
            image_h5 = h5py.File(os.path.join(self.IMG_FEAT_DIR, '{}.h5'.format(image_id)), 'r')
            scale = image_h5['im_info'][0, 2]
            image_feat = torch.tensor(image_h5['head'])  # [1, 1024, ih, iw]
            # RoI-pool positive M-RCNN detections
            det_box = self.img_to_det_box[image_id].reshape(-1, 81, 4)  # [300, 81, 4]
            det_score = self.img_to_det_score[image_id]                 # [300, 81]
            det_box = np.transpose(det_box[:, 1:], axes=[1, 0, 2])      # [80, 300, 4]
            det_score = np.transpose(det_score[:, 1:], axes=[1, 0])     # [80, 300]

            this_thresh = np.sort(det_score, axis=None)[-self.top_N]
            positive = det_score >= this_thresh        # [80, 300]
            pos_box = torch.tensor(det_box[positive])  # [*, 4]
            pos_score = torch.tensor(det_score[positive], device=self.device)  # [*]
            cls_num_list = np.sum(positive, axis=1).tolist()                   # [80]

            pos_feat = mrcn_crop_pool_layer(image_feat, pos_box * scale)  # [*, 1024, 7, 7]
            pos_feat = pos_feat.to(self.device).unsqueeze(0)  # [1, *, 1024, 7, 7]
            pos_box = pos_box.to(self.device)
            for exp_id, tokens in exps:
                # Load word feature
                assert isinstance(tokens, list)
                sent_feat = torch.tensor(self.idx_to_glove[tokens], device=self.device)
                sent_feat = sent_feat.unsqueeze(0)  # [1, *, 300]
                yield exp_id, pos_feat, sent_feat, pos_box, pos_score, cls_num_list

    def __len__(self):
        return len(self.refs)


def _test():
    import json
    from tqdm import tqdm
    refdb = json.load(open('cache/refdb_refcoco_unc_nopos.json', 'r'))
    ctxdb = json.load(open('cache/ctxdb_refcoco_unc.json', 'r'))
    dataset = DetBoxDataset(refdb, ctxdb, 'train')
    neg_num, pos_num, total_num = [], [], []
    for pos_rois, neg_rois in tqdm(dataset, ascii=True):
        neg_num.append(len(neg_rois))
        pos_num.append(len(pos_rois))
        total_num.append(len(pos_rois) + len(neg_rois))
    print('neg min: {}, neg max: {}, neg mean: {}'.format(min(neg_num), max(neg_num), sum(neg_num) / len(neg_num)))
    print('pos min: {}, pos max: {}, pos mean: {}'.format(min(pos_num), max(pos_num), sum(pos_num) / len(pos_num)))
    print('total min: {}, total max: {}, total mean: {}'.format(min(total_num), max(total_num), sum(total_num) / len(total_num)))


if __name__ == '__main__': _test()
