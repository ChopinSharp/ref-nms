import json

import numpy as np
import spacy
from tqdm import tqdm

from lib.refer import REFER
from utils.constants import CAT_ID_TO_NAME, EVAL_SPLITS_DICT
from utils.misc import xywh_to_xyxy, calculate_iou


POS_OF_INTEREST = {'NOUN', 'NUM', 'PRON', 'PROPN'}


def load_glove_feats():
    glove_path = 'data/glove.840B.300d.txt'
    print('loading GloVe feature from {}'.format(glove_path))
    glove_dict = {}
    with open(glove_path, 'r') as f:
        with tqdm(total=2196017, desc='Loading GloVe', ascii=True) as pbar:
            for line in f:
                tokens = line.split(' ')
                assert len(tokens) == 301
                word = tokens[0]
                vec = list(map(lambda x: float(x), tokens[1:]))
                glove_dict[word] = vec
                pbar.update(1)
    return glove_dict


def cosine_similarity(feat_a, feat_b):
    return np.sum(feat_a * feat_b) / np.sqrt(np.sum(feat_a * feat_a) * np.sum(feat_b * feat_b))


def build_ctxdb(dataset, split_by):
    dataset_splitby = '{}_{}'.format(dataset, split_by)
    # Load refer
    refer = REFER('data/refer', dataset, split_by)
    # Load GloVe feature
    glove_dict = load_glove_feats()
    # Construct COCO category GloVe feature
    cat_id_to_glove = {}
    for cat_id, cat_name in CAT_ID_TO_NAME.items():
        cat_id_to_glove[cat_id] = [np.array(glove_dict[t], dtype=np.float32) for t in cat_name.split(' ')]
    # Spacy to extract POS tags
    nlp = spacy.load('en_core_web_sm')
    # Go through the refdb
    ctxdb = {}
    for split in (['train'] + EVAL_SPLITS_DICT[dataset_splitby]):
        exp_to_ctx = {}
        gt_miss_num, empty_num, sent_num = 0, 0, 0
        coco_box_num_list, ctx_box_num_list = [], []
        ref_ids = refer.getRefIds(split=split)
        for ref_id in tqdm(ref_ids, ascii=True, desc=split):
            ref = refer.Refs[ref_id]
            image_id = ref['image_id']
            gt_box = xywh_to_xyxy(refer.Anns[ref['ann_id']]['bbox'])
            gt_cat = refer.Anns[ref['ann_id']]['category_id']
            for sent in ref['sentences']:
                sent_num += 1
                sent_id = sent['sent_id']
                doc = nlp(sent['sent'])
                noun_tokens = [token.text for token in doc if token.pos_ in POS_OF_INTEREST]
                # print('SENT', sent['sent'])
                # print('NOUN TOKENS', noun_tokens)
                noun_glove_list = [np.array(glove_dict[t], dtype=np.float32) for t in noun_tokens if t in glove_dict]
                gt_hit = False
                ctx_list = []
                for ann in refer.imgToAnns[image_id]:
                    ann_glove_list = cat_id_to_glove[ann['category_id']]
                    cos_sim_list = [cosine_similarity(ann_glove, noun_glove)
                                    for ann_glove in ann_glove_list
                                    for noun_glove in noun_glove_list]
                    # print(CAT_ID_TO_NAME[ann['category_id']], cos_sim_list)
                    max_cos_sim = max(cos_sim_list, default=0.)
                    if max_cos_sim > 0.4:
                        ann_box = xywh_to_xyxy(ann['bbox'])
                        if calculate_iou(ann_box, gt_box) > 0.9:
                            gt_hit = True
                        else:
                            ctx_list.append({'box': ann_box, 'cat_id': ann['category_id']})
                if not gt_hit:
                    gt_miss_num += 1
                if not ctx_list:
                    empty_num += 1
                exp_to_ctx[sent_id] = {'gt': {'box': gt_box, 'cat_id': gt_cat}, 'ctx': ctx_list}
                coco_box_num_list.append(len(refer.imgToAnns[image_id]))
                ctx_box_num_list.append(len(ctx_list) + 1)
        print('GT miss: {} out of {}'.format(gt_miss_num, sent_num))
        print('empty ctx: {} out of {}'.format(empty_num, sent_num))
        print('COCO box per sentence: {:.3f}'.format(sum(coco_box_num_list) / len(coco_box_num_list)))
        print('ctx box per sentence: {:.3f}'.format(sum(ctx_box_num_list) / len(ctx_box_num_list)))
        ctxdb[split] = exp_to_ctx
    # Save results
    save_path = 'cache/std_ctxdb_{}.json'.format(dataset_splitby)
    print('saving ctxdb to {}'.format(save_path))
    with open(save_path, 'w') as f:
        json.dump(ctxdb, f)


def main():
    print('building ctxdb...')
    for dataset, split_by in [('refcoco', 'unc'), ('refcoco+', 'unc'), ('refcocog', 'umd')]:
        print('building {}_{}...'.format(dataset, split_by))
        build_ctxdb(dataset, split_by)
    print()


main()
