import pickle
import json
from time import time


# def _vocab_stat(dataset='refcoco', split_by='unc'):
#     from lib.refer import REFER
#     import numpy as np
#     import spacy
#     from tqdm import tqdm
#     refer = REFER('data/refer', dataset=dataset, splitBy=split_by)
#     with open('cache/vocab_{}_{}.txt'.format(dataset, split_by), 'r') as f:
#         vocab = [wd[:-1] for wd in f.readlines()]
#     # filter with glove
#     print('filtering with glove...')
#     filtered_token_length = []
#     zero_len_refer_num = 0
#     for ref in tqdm(refer.Refs.values()):
#         total_length = 0
#         for sent in ref['sentences']:
#             length = len([wd for wd in sent['tokens'] if wd in vocab])
#             total_length += length
#             filtered_token_length.append(length)
#         if total_length == 0: zero_len_refer_num += 1
#     filtered_token_length = np.array(filtered_token_length, dtype=np.float32)
#     print('max={:.2f}, min={:.2f}, mean={:.2f}, std={:.2f}'
#           .format(filtered_token_length.max(), filtered_token_length.min(),
#                   filtered_token_length.mean(), filtered_token_length.std()))
#     print('refer with no valid sentence: {}'.format(zero_len_refer_num))
#     # filter with POS and glove
#     print('filtering with POS and glove...')
#     nlp = spacy.load('en_core_web_sm')
#     POS = {'ADJ', 'ADV', 'NOUN', 'NUM', 'PRON', 'PROPN', 'VERB'}
#     filtered_token_length = []
#     zero_len_ref = []
#     for ref_id, ref in tqdm(refer.Refs.items()):
#         total_length = 0
#         for sent in ref['sentences']:
#             doc = nlp(sent['sent'])
#             tokens = [token.text for token in doc if token.pos_ in POS]
#             length = len([wd for wd in tokens if wd in vocab])
#             total_length += length
#             filtered_token_length.append(length)
#         if total_length == 0: zero_len_ref.append(ref_id)
#     filtered_token_length = np.array(filtered_token_length, dtype=np.float32)
#     print('max={:.2f}, min={:.2f}, mean={:.2f}, std={:.2f}'
#           .format(filtered_token_length.max(), filtered_token_length.min(),
#                   filtered_token_length.mean(), filtered_token_length.std()))
#     print('refer with no valid sentence: {}'.format(zero_len_ref))


def _disassemble_proposals():
    RPN_PROPOSAL_FILE = 'data/res101_coco_minus_refer_notime_proposals.pkl'
    with open(RPN_PROPOSAL_FILE, 'rb') as f:
        proposals = pickle.load(f, encoding='latin1')
    img_to_rois = {}
    img_to_box_scores = {}
    img_to_roi_scores = {}
    img_to_boxes = {}
    for p in proposals:
        image_id = p['image_id']
        img_to_rois[image_id] = p['rois']
        img_to_box_scores[image_id] = p['scores']
        img_to_roi_scores[image_id] = p['roi_scores']
        img_to_boxes[image_id] = p['boxes']
    print('saving rois...')
    pickle.dump(img_to_rois, open('cache/rpn_rois.pkl', 'wb'))
    print('saving scores...')
    pickle.dump(img_to_box_scores, open('cache/rpn_box_scores.pkl', 'wb'))
    print('saving roi scores...')
    pickle.dump(img_to_roi_scores, open('cache/rpn_roi_scores.pkl', 'wb'))
    print('saving boxes...')
    pickle.dump(img_to_boxes, open('cache/rpn_boxes.pkl', 'wb'))


def _save_raw_scores():
    RAW_SCORE_FILE = 'data/raw_scores.pkl'
    with open(RAW_SCORE_FILE, 'rb') as f:
        raw_scores = pickle.load(f, encoding='latin1')
    img_to_raw_score = {}
    for sc in raw_scores:
        img_to_raw_score[sc['image_id']] = sc['raw_scores']
    pickle.dump(img_to_raw_score, open('cache/raw_scores.pkl', 'wb'))


def _sanity_check(gpu_id):
    from lib.predictor import LTRPredictor
    from tools.train_ltr import init_ltr_predictor
    import pickle
    import h5py
    import torch
    from utils.misc import mrcn_crop_pool_layer
    device = torch.device('cpu')#('cuda', gpu_id) # 0.2975 # 230246
    predictor = LTRPredictor()
    init_ltr_predictor(predictor)
    predictor.to(device)
    predictor.eval()
    rois = pickle.load(open('cache/rpn_rois.pkl', 'rb'))
    # k = next(iter(rois.keys()))
    max_num = 500
    cnt = 0
    tic = time()
    for k in rois.keys():
        cnt += 1
        roi = rois[k]
        image_h5 = h5py.File('cache/head_feats/matt-mrcn/{}.h5'.format(k), 'r')
        scale = image_h5['im_info'][0, 2]
        image_feat = torch.tensor(image_h5['head']).to(device)
        roi = torch.tensor(roi).to(device) * scale
        roi_feat = mrcn_crop_pool_layer(image_feat, roi)
        # head_feat = predictor.head_net(roi_feat).mean(dim=(2,3))
        # cls_score = predictor.cls_score_net(head_feat)
        # cls_score = torch.nn.functional.softmax(cls_score, dim=1)
        # scores = pickle.load(open('cache/rpn_box_scores.pkl', 'rb'))
        # score = torch.tensor(scores[k]).to(device)
        # torch.save(roi_feat, 'temp/{}.pth'.format(k))
        # print(score)
        # print(cls_score)
        print('======================', cnt, k, '========================')
        # print('distance:', ((cls_score - score)**2).sum().item())
        # print('max allocated:', torch.cuda.memory_allocated(device) / (2**20))
        # print('max cached:', torch.cuda.memory_cached(device) / (2**20))
        if cnt == max_num:
            break
    print('time per image:', (time() - tic) / max_num)


def _split_refdb():
    with open('cache/refdb_refcoco_unc_nopos.json', 'r') as f:
        refdb = json.load(f)
    val = [{'image_id': ref['image_id'], 'exp_id': ref['exp_id']} for ref in refdb['val']]
    cut = len(val) // 2
    valA = val[:cut]
    valB = val[cut:]
    testA = [{'image_id': ref['image_id'], 'exp_id': ref['exp_id']} for ref in refdb['testA']]
    testB = [{'image_id': ref['image_id'], 'exp_id': ref['exp_id']} for ref in refdb['testB']]
    refcoco_unc_eval_index = {'valA': valA, 'valB': valB, 'testA': testA, 'testB': testB}
    with open('cache/refcoco_unc_eval_index.json', 'w') as f:
        json.dump(refcoco_unc_eval_index, f)


if __name__ == '__main__':
    # _disassemble_proposals()
    # _sanity_check()
    # _split_refdb()
    _sanity_check(2)