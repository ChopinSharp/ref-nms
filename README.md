# Ref-NMS
Official codebase for AAAI 2021 paper ["Ref-NMS: Breaking Proposal Bottlenecks in Two-Stage Referring Expression Grounding"](https://arxiv.org/abs/2009.01449).

## Prerequisites
The following dependencies should be enough. See [environment.yml](environment.yml) for complete environment settings.
- python 3.7.6
- pytorch 1.1.0
- torchvision 0.3.0
- tensorboard 2.1.0
- spacy 2.2.3

## Data Preparation
Follow instructions in `data/README.md` to setup `data` directory. 

Run following script to setup `cache` directory:
```
sh scripts/prepare_data.sh
```
This should generate following files under `cache` directory:
- vocabulary file: `std_vocab_<dataset>_<split_by>.txt`
- selected GloVe feature: `std_glove_<dataset>_<split_by>.npy`
- referring expression database: `std_refdb_<dataset>_<split_by>.json`
- critical objects database: `std_ctxdb_<dataset>_<split_by>.json`


## Train
**Train with binary XE loss:**
```
PYTHONPATH=$PWD python tools/train_att_vanilla.py --dataset refcoco --split-by unc
```

**Train with ranking loss:**
```
PYTHONPATH=$PWD python tools/train_att_rank.py --dataset refcoco --split-by unc
```

We use tensorboard to monitor the training process. The log file can be found in `tb` folder.

## Evaluate Recall
**Save Ref-NMS proposals:**
```
PYTHONPATH=$PWD python tools/save_ref_nms_proposals.py --dataset refcoco --split-by unc --tid <tid> --m <loss_type>
```
`<loss_type>` can be either `att_vanilla` for binary XE loss or `att_rank` for rank loss.

**Evaluate recall on referent object:**
```
PYTHONPATH=$PWD python tools/eval_proposal_hit_rate.py --m <loss_type> --dataset refcoco --split-by unc --tid <tid> --conf <conf>
```
`conf` parameter is the score threshold used to filter Ref-NMS proposals. It should be picked properly so that the recall of the referent is high while the number of proposals per expression is around 8-10.

**Evaluate recall on critical objects:**
```
PYTHONPATH=$PWD python tools/eval_proposal_ctx_recall.py --m <loss_type> --dataset refcoco --split-by unc --tid <tid> --conf <conf>
```

## Evaluate REG Performance
Save MAttNet-style detection file:
```
PYTHONPATH=$PWD python tools/save_matt_dets.py --dataset refcoco --split-by unc --m <loss_type> --tid <tid> --conf <conf>
```
This script will save all the detection information needed for downstream REG evaluation to `output/matt_dets_<loss_type>_<tid>_<dataset>_<split_by>_<top_N>.json`.

We provide altered version of [MAttNet](https://github.com/ChopinSharp/MAttNet) and [CM-A-E](https://github.com/ChopinSharp/CM-Erase-REG) for downstream REG task evaluation. 

First, follow the README in each repository to reproduce the original reported results as baseline (c.f. Table 2 in our paper). Then, run the following commands to evaluate on REC and RES task:
```
# Evaluate REC performance
python tools/extract_mrcn_ref_feats.py --dataset refcoco --splitBy unc --tid <tid> --top-N 0 --m <loss_type>
python tools/eval_ref.py --dataset refcoco --splitBy unc --tid <tid> --top-N 0 --m <loss_type>
# Evaluate RES performance
python tools/run_propose_to_mask.py --dataset refcoco --splitBy unc --tid <tid> --top-N 0 --m <loss_type>
python tools/eval_ref_masks.py --dataset refcoco --splitBy unc --tid <tid> --top-N 0 --m <loss_type> --save
```

## Pretrained Models
We provide pre-trained model weights as long as the corresponding **MAttNet-style detection file** (note the MattNet-style detection files can be directly used to evaluate downstream REG task performance). With these files, one can easily reproduce our reported results.

[[Google Drive]](https://drive.google.com/drive/folders/1BPqWW0LrAEBFna7b-ORF2TcrY7K_DDvM?usp=sharing) [[Baidu Disk]](https://pan.baidu.com/s/1G4k7APKSUs-_5StXoYaNrA) (extraction code: 5a9r)

## Citation
```
@inproceedings{chen2021ref,
  title={Ref-NMS: Breaking Proposal Bottlenecks in Two-Stage Referring Expression Grounding},
  author={Chen, Long and Ma, Wenbo and Xiao, Jun and Zhang, Hanwang and Chang, Shih-Fu},
  booktitle={AAAI},
  year={2021}
}
```
