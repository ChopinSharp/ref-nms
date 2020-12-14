from argparse import ArgumentParser
import json
import os
from time import time
import copy
import itertools

from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.init import zeros_, xavier_uniform_
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence

from lib.rank_utils import RankDataset
from lib.predictor import AttVanillaPredictorV2
from utils.misc import get_time_id


PRETRAINED_MRCN = 'data/res101_mask_rcnn_iter_1250000_cpu.pth'

CONFIG = dict(
    HEAD_LR=2e-4,
    HEAD_WD=1e-3,
    REF_LR=5e-4,
    REF_WD=1e-3,
    RNN_LR=5e-4,
    RNN_WD=1e-3,
    BATCH_SIZE=8,
    EPOCH=4,
    ATT_DROPOUT_P=0.5,
    RANK_DROPOUT_P=0.5,
    LOSS_MARGIN=0.1,
    TOP_H=100,
    ROI_PER_LEVEL=10,
    NEGATIVE_NUM=150,
    LEVEL_NUM=6
)
LOG_INTERVAL = 50
VAL_INTERVAL = 1000


class RankLoss:

    def __init__(self, margin, device):
        self.margin = margin
        self.zero_scalar = torch.tensor(0., device=device)

    def __call__(self, rank_score, sampled_pairs):
        """Compute Hinge loss on sampled pairs.

        Args:
            rank_score: Tensor of shape [batch_size, roi_num].
            sampled_pairs: Tensor of shape [batch_size, pair_num, 2]

        Returns:
            loss: Computed loss.

        """
        pos_indices = sampled_pairs[:, :, 0]  # [batch_size, pair_num]
        neg_indices = sampled_pairs[:, :, 1]  # [batch_size, pair_num]
        pos_scores = torch.gather(rank_score, 1, pos_indices)  # [batch_size, pair_num]
        neg_scores = torch.gather(rank_score, 1, neg_indices)  # [batch_size, pair_num]
        loss = torch.max(neg_scores - pos_scores + self.margin, self.zero_scalar).mean()
        return loss


class PairSampler:

    def __init__(self, level_num, roi_per_level, negative_num, top_h, device):
        self.level_num = level_num
        self.roi_per_level = roi_per_level
        self.negative_num = negative_num
        self.top_h = top_h
        self.device = device

    def __call__(self, rank_score):
        """Sample training pairs with hard negative mining.

        Args:
            rank_score: Tensor of shape [batch_size, roi_num].

        Returns:
            batch_sampled_pairs: Tensor of shape [batch_size, pair_num, 2].
                `batch_sampled_pairs[:, :, 0]` are indices of positive ROIs,
                `batch_sampled_pairs[:, :, 1]` are indices of negative ROIs.

        """
        N, R = rank_score.size()
        assert R == self.negative_num + (self.level_num - 1) * self.roi_per_level
        batch_sorted_idx = rank_score.argsort(dim=1, descending=True)  # [batch_size, roi_num]
        batch_sampled_pairs = []
        for b in range(N):
            sorted_idx_list = batch_sorted_idx[b].tolist()
            pair_list = []
            for l in range(self.level_num - 1):
                start_idx = self.negative_num + l * self.roi_per_level
                pos_idx = [i for i in range(start_idx, start_idx + self.roi_per_level)]
                neg_idx = [i for i in sorted_idx_list if i in range(start_idx)]
                neg_idx = neg_idx[:self.top_h]
                pair_list.extend(itertools.product(pos_idx, neg_idx))
            batch_sampled_pairs.append(pair_list)
        batch_sampled_pairs = torch.tensor(batch_sampled_pairs, device=self.device)
        return batch_sampled_pairs


def init_att_vanilla_predictor(predictor):
    # Load pre-trained weights from M-RCNN
    mrcn_weights = torch.load(PRETRAINED_MRCN)
    c4_weights = {
        k[len('resnet.layer4.'):]: v
        for k, v in mrcn_weights.items()
        if k.startswith('resnet.layer4')
    }
    assert len(c4_weights) == 50
    predictor.head.load_state_dict(c4_weights)
    # Initialize new layers
    count = 0
    for name, param in predictor.named_parameters():
        if 'head' in name:
            continue
        if 'weight' in name:
            xavier_uniform_(param)
            count += 1
        elif 'bias' in name:
            zeros_(param)
            count += 1
    assert count == 20


def compute_loss(predictor, sampler, criterion, device, enable_grad, roi_feats, word_feats, sent_len):
    with torch.autograd.set_grad_enabled(enable_grad):
        roi_feats = roi_feats.to(device)
        word_feats = word_feats.to(device)
        packed_sent_feats = pack_padded_sequence(word_feats, sent_len, enforce_sorted=False, batch_first=True)
        scores, *_ = predictor.forward(roi_feats, packed_sent_feats)
        sigmoid_scores = torch.sigmoid(scores)
        pairs = sampler(sigmoid_scores)
        loss = criterion(sigmoid_scores, pairs)
    return loss, sigmoid_scores


def main(args):
    if args.resume is None:
        tid = get_time_id()
        start_epoch = 0
    else:
        *_, tid, start_epoch = args.resume[:-4].split('_')
        tid += '_cont'
        start_epoch = int(start_epoch)
    if args.epoch is not None:
        CONFIG['EPOCH'] = args.epoch
    dataset_splitby = '{}_{}'.format(args.dataset, args.split_by)
    device = torch.device('cuda', args.gpu_id)
    refdb_path = 'cache/std_refdb_{}.json'.format(dataset_splitby)
    print('loading refdb from {}...'.format(refdb_path))
    with open(refdb_path, 'r') as f:
        refdb = json.load(f)
    ctxdb_path = 'cache/std_ctxdb_{}.json'.format(dataset_splitby)
    print('loading ctxdb from {}...'.format(ctxdb_path))
    with open(ctxdb_path, 'r') as f:
        ctxdb = json.load(f)
    # Build dataloaders
    dataset_settings = dict(level_num=CONFIG['LEVEL_NUM'], roi_per_level=CONFIG['ROI_PER_LEVEL'],
                            negative_num=CONFIG['NEGATIVE_NUM'])
    trn_dataset = RankDataset(refdb, ctxdb, 'train', **dataset_settings)
    val_dataset = RankDataset(refdb, ctxdb, 'val', **dataset_settings)
    loader_settings = dict(batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=4)
    trn_loader = DataLoader(trn_dataset, **loader_settings)
    val_loader = DataLoader(val_dataset, drop_last=True, **loader_settings)
    # Tensorboard writer
    tb_dir = 'tb/att_rank/{}'.format(tid)
    trn_wrt = SummaryWriter(os.path.join(tb_dir, 'train'))
    val_wrt = SummaryWriter(os.path.join(tb_dir, 'val'))
    sc_wrts = {l: SummaryWriter(os.path.join(tb_dir, 'level:{}'.format(l))) for l in range(CONFIG['LEVEL_NUM'])}
    # Build and init predictor
    predictor = AttVanillaPredictorV2(att_dropout_p=CONFIG['ATT_DROPOUT_P'], rank_dropout_p=CONFIG['RANK_DROPOUT_P'])
    init_att_vanilla_predictor(predictor)
    predictor.to(device)
    # Setup pair sampler
    sampler = PairSampler(CONFIG['LEVEL_NUM'], CONFIG['ROI_PER_LEVEL'], CONFIG['NEGATIVE_NUM'], CONFIG['TOP_H'], device)
    # Setup loss
    criterion = RankLoss(CONFIG['LOSS_MARGIN'], device)
    # Setup optimizer
    ref_params = list(predictor.att_fc.parameters()) + list(predictor.rank_fc.parameters()) \
                 + list(predictor.vis_a_fc.parameters()) + list(predictor.vis_r_fc.parameters())
    ref_optimizer = optim.Adam(ref_params, lr=CONFIG['REF_LR'], weight_decay=CONFIG['REF_WD'])
    rnn_optimizer = optim.Adam(predictor.rnn.parameters(), lr=CONFIG['RNN_LR'], weight_decay=CONFIG['RNN_WD'])
    head_optimizer = optim.Adam(predictor.head.parameters(), lr=CONFIG['HEAD_LR'], weight_decay=CONFIG['HEAD_WD'])
    common_args = dict(mode='min', factor=0.6, verbose=True, threshold_mode='rel', patience=3)
    ref_scheduler = optim.lr_scheduler.ReduceLROnPlateau(ref_optimizer, min_lr=CONFIG['REF_LR']/100, **common_args)
    rnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(rnn_optimizer, min_lr=CONFIG['RNN_LR']/100, **common_args)
    head_scheduler = optim.lr_scheduler.ReduceLROnPlateau(head_optimizer, min_lr=CONFIG['HEAD_LR']/100, **common_args)
    # Start training
    if args.resume is not None:
        resume_ckpt = torch.load(os.path.join('output', args.resume))
        predictor.load_state_dict(resume_ckpt['model'])
        ref_optimizer.load_state_dict(resume_ckpt['ref_optimizer'])
        rnn_optimizer.load_state_dict(resume_ckpt['rnn_optimizer'])
        head_optimizer.load_state_dict(resume_ckpt['head_optimizer'])
        ref_scheduler.load_state_dict(resume_ckpt['ref_scheduler'])
        rnn_scheduler.load_state_dict(resume_ckpt['rnn_scheduler'])
        head_scheduler.load_state_dict(resume_ckpt['head_scheduler'])
    step = 0
    trn_running_loss = 0.
    best_model = {'avg_val_loss': float('inf'), 'epoch': None, 'step': None, 'weights': None}
    tic = time()
    for epoch in range(start_epoch, CONFIG['EPOCH']):
        for trn_batch in trn_loader:
            # Train for one step
            step += 1
            predictor.train()
            trn_loss, _ = compute_loss(predictor, sampler, criterion, device, True, *trn_batch)
            head_optimizer.zero_grad()
            ref_optimizer.zero_grad()
            rnn_optimizer.zero_grad()
            trn_loss.backward()
            head_optimizer.step()
            ref_optimizer.step()
            rnn_optimizer.step()
            trn_running_loss += trn_loss.item()
            # Log training loss
            if step % LOG_INTERVAL == 0:
                avg_trn_loss = trn_running_loss / LOG_INTERVAL
                print('[TRN Loss] epoch {} step {}: {:.6f}'.format(epoch + 1, step, avg_trn_loss))
                trn_wrt.add_scalar('loss', avg_trn_loss, step)
                trn_running_loss = 0.
            # Eval on whole val split
            if step % VAL_INTERVAL == 0:
                # Compute and log val loss
                predictor.eval()
                val_loss_list = []
                level_score_mean_list = {l: [] for l in range(CONFIG['LEVEL_NUM'])}
                pbar = tqdm(total=len(val_dataset), ascii=True, desc='computing val loss')
                for val_batch in val_loader:
                    val_loss, val_score = compute_loss(predictor, sampler, criterion, device, False, *val_batch)
                    val_loss_list.append(val_loss.item())
                    neg_rank_score = val_score[:, :CONFIG['NEGATIVE_NUM']]
                    level_score_mean_list[0].append(neg_rank_score.mean().item())
                    pos_rank_score = val_score[:, CONFIG['NEGATIVE_NUM']:]
                    pos_rank_score = pos_rank_score.reshape(CONFIG['BATCH_SIZE'], CONFIG['LEVEL_NUM'] - 1, -1)
                    for l in range(CONFIG['LEVEL_NUM'] - 1):
                        level_score_mean_list[l + 1].append(pos_rank_score[:, l].mean().item())
                    pbar.update(val_batch[0].size(0))
                pbar.close()
                avg_val_loss = sum(val_loss_list) / len(val_loss_list)
                print('[VAL Loss] epoch {} step {}: {:.6f}'.format(epoch + 1, step, avg_val_loss))
                val_wrt.add_scalar('loss', avg_val_loss, step)
                for l in range(CONFIG['LEVEL_NUM']):
                    avg_score_mean = sum(level_score_mean_list[l]) / len(level_score_mean_list[l])
                    sc_wrts[l].add_scalar('rank score mean', avg_score_mean, step)
                # Update learning rate
                head_scheduler.step(avg_val_loss)
                ref_scheduler.step(avg_val_loss)
                rnn_scheduler.step(avg_val_loss)
                # Track model with lowest val loss
                if avg_val_loss < best_model['avg_val_loss']:
                    best_model['avg_val_loss'] = avg_val_loss
                    best_model['epoch'] = epoch + 1
                    best_model['step'] = step
                    best_model['weights'] = copy.deepcopy(predictor.state_dict())
        # Save checkpoint after each epoch
        epoch_ckpt = {
            'ref_optimizer': ref_optimizer.state_dict(),
            'rnn_optimizer': rnn_optimizer.state_dict(),
            'head_optimizer': head_optimizer.state_dict(),
            'ref_scheduler': ref_scheduler.state_dict(),
            'rnn_scheduler': rnn_scheduler.state_dict(),
            'head_scheduler': head_scheduler.state_dict(),
            'model': predictor.state_dict()
        }
        save_path = 'output/att_rank_ckpt_{}_{}.pth'.format(tid, epoch + 1)
        torch.save(epoch_ckpt, save_path)
    # Save best model
    time_spent = int(time() - tic) // 60
    print('\nTraining completed in {} h {} m.'.format(time_spent // 60, time_spent % 60))
    print('Found model with lowest val loss at epoch {epoch} step {step}.'.format(**best_model))
    save_path = 'output/att_rank_{}_b.pth'.format(tid)
    torch.save(best_model['weights'], save_path)
    print('Saved best model weights to {}'.format(save_path))
    # Close summary writer
    trn_wrt.close()
    val_wrt.close()
    for wrt in sc_wrts.values():
        wrt.close()
    # Log training procedure
    model_info = {
        'type': 'att_rank',
        'dataset': dataset_splitby,
        'config': CONFIG
    }
    with open('output/att_rank_{}.json'.format(tid), 'w') as f:
        json.dump(model_info, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='refcoco')
    parser.add_argument('--split-by', default='unc')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    main(parser.parse_args())
