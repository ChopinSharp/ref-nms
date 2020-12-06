import argparse
import pickle
import json

from utils.constants import EVAL_SPLITS_DICT, COCO_CAT_NAMES, CAT_NAME_TO_ID
from lib.refer import REFER


def threshold_with_top_N(exp_to_proposals, top_N):
    results = {}
    for exp_id, proposals in exp_to_proposals.items():
        assert len(proposals) >= 1
        results[exp_id] = sorted(proposals, key=lambda p: p['score'], reverse=True)[:top_N]
    return results


def threshold_with_confidence(exp_to_proposals, conf):
    results = {}
    for exp_id, proposals in exp_to_proposals.items():
        assert len(proposals) >= 1
        sorted_proposals = sorted(proposals, key=lambda p: p['score'], reverse=True)
        thresh_proposals = [sorted_proposals[0]]
        for prop in sorted_proposals[1:]:
            if prop['score'] > conf:
                thresh_proposals.append(prop)
            else:
                break
        results[exp_id] = thresh_proposals
    return results


def main(args):
    # Setup
    assert args.top_N is None or args.conf is None
    assert args.top_N is not None or args.conf is not None
    dataset_splitby = '{}_{}'.format(args.dataset, args.split_by)
    refer = REFER('data/refer', dataset=args.dataset, splitBy=args.split_by)
    det_id = 0
    matt_dets = []
    eval_splits = EVAL_SPLITS_DICT[dataset_splitby]

    # Add model detections for valid sentences
    proposal_path = 'cache/proposals_{}_{}_{}.pkl'.format(args.m, args.dataset, args.tid)
    print('loading proposals from {}...'.format(proposal_path))
    with open(proposal_path, 'rb') as f:
        proposal_dict = pickle.load(f)
    for split in eval_splits:
        exp_to_proposals = proposal_dict[split]
        if args.top_N is not None:
            exp_to_proposals = threshold_with_top_N(exp_to_proposals, args.top_N)
        if args.conf is not None:
            exp_to_proposals = threshold_with_confidence(exp_to_proposals, args.conf)
        for exp_id, proposals in exp_to_proposals.items():
            ref = refer.sentToRef[exp_id]
            ref_id = ref['ref_id']
            image_id = ref['image_id']
            for proposal in proposals:
                x1, y1, x2, y2 = proposal['box']
                w, h = x2 - x1, y2 - y1
                box = (x1, y1, w, h)
                cat_name = COCO_CAT_NAMES[proposal['cls_idx']]
                det = {
                    'det_id': det_id,
                    'h5_id': det_id,
                    'ref_id': ref_id,
                    'sent_id': exp_id,
                    'image_id': image_id,
                    'box': box,
                    'category_id': CAT_NAME_TO_ID[cat_name],
                    'category_name': cat_name,
                    'split': split,
                    # 'cls_score': proposal['det_score'],
                    # 'rank_score': proposal['rank_score'],
                    # 'fin_score': proposal['score']
                }
                matt_dets.append(det)
                det_id += 1

    # Print out stats and save detections
    for split in eval_splits:
        exp_num = len({det['sent_id'] for det in matt_dets if det['split'] == split})
        det_num = len([det for det in matt_dets if det['split'] == split])
        print('[{:5s}] {} / {} = {:.2f} detections per expression'
              .format(split, det_num, exp_num, det_num / exp_num))
    top_N = 0 if args.top_N is None else args.top_N
    save_path = 'output/matt_dets_{}_{}_{}_{}.json'.format(args.m, args.tid, dataset_splitby, top_N)
    # save_path = 'output/matt_dets_{}_{}_{}_{}_more.json'.format(args.m, args.tid, dataset_splitby, top_N)
    print('saving detections to {}...'.format(save_path))
    with open(save_path, 'w') as f:
        json.dump(matt_dets, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='refcoco')
    parser.add_argument('--split-by', type=str, default='unc')
    parser.add_argument('--m', type=str, required=True)
    parser.add_argument('--top-N', type=int, default=None)
    parser.add_argument('--tid', type=str, required=True)
    parser.add_argument('--conf', type=float, default=None)
    main(parser.parse_args())
