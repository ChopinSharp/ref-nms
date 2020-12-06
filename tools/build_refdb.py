import json

import numpy as np

from lib.refer import REFER
from utils.misc import xywh_to_xyxy


DATASET_SPLITS = {
    'refcoco_unc': ['train', 'val', 'testA', 'testB'],
    'refcoco+_unc': ['train', 'val', 'testA', 'testB'],
    'refcocog_umd': ['train', 'val', 'test']
}


def build_refdb(dataset, split_by):
    # Load refer data
    refer = REFER('data/refer', dataset, split_by)
    # Load vocab
    with open('cache/std_vocab_{}_{}.txt'.format(dataset, split_by)) as f:
        idx_to_wd = [wd[:-1] for wd in f.readlines()]  # trim off newline
    wd_to_idx = {}
    for idx, wd in enumerate(idx_to_wd):
        wd_to_idx[wd] = idx
    # Build refdb
    dataset_splitby = '{}_{}'.format(dataset, split_by)
    data = {'dataset_splitby': dataset_splitby}
    for split in DATASET_SPLITS[dataset_splitby]:
        split_data = []
        for ref_id in refer.getRefIds(split=split):
            ref = refer.Refs[ref_id]
            image_id = ref['image_id']
            ann_id = ref['ann_id']
            ann = refer.Anns[ann_id]
            bbox = xywh_to_xyxy(ann['bbox'])
            # Filter with POS
            for sent in ref['sentences']:
                sent_id, tokens = sent['sent_id'], sent['tokens']
                # Encode with vocab
                encoded_tokens = [wd_to_idx[wd] if wd in wd_to_idx else 0 for wd in tokens]
                split_data.append({
                    'exp_id': sent_id,
                    'ref_id': ref_id,
                    'image_id': image_id,
                    'bbox': bbox,
                    'tokens': encoded_tokens
                })
        data[split] = split_data
    # Print out statistics
    print('STATS for {}'.format(dataset_splitby))
    for split in DATASET_SPLITS[dataset_splitby]:
        ref_num = len({ref['ref_id'] for ref in data[split]})
        sent_num = len(data[split])
        avg_sent_num = sent_num / ref_num
        token_len = np.array([len(ref['tokens']) for ref in data[split]], dtype=np.float32)
        print('[{}]'.format(split))
        print('ref_num={}, avg_sent_num={:.4f}'.format(ref_num, avg_sent_num))
        print('token_len: mean={:.4f}, std={:.4f}'.format(token_len.mean(), token_len.std()))
    # Save refdb to json file
    refdb_save_path = 'cache/std_refdb_{}.json'.format(dataset_splitby)
    print('saving refdb to file: {}'.format(refdb_save_path))
    with open(refdb_save_path, 'w') as f:
        json.dump(data, f)


def main():
    print('building refdb...')
    for dataset, split_by in [('refcoco', 'unc'), ('refcoco+', 'unc'), ('refcocog', 'umd')]:
        print('building {}_{}...'.format(dataset, split_by))
        build_refdb(dataset, split_by)
    print()


main()
