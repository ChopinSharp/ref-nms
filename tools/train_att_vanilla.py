from argparse import ArgumentParser
import json
import os
from time import time
import copy

from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.init import zeros_, xavier_uniform_
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence

from lib.vanilla_utils import DetBoxDataset
from lib.predictor import AttVanillaPredictorV2
from utils.misc import get_time_id


PRETRAINED_MRCN = 'data/res101_mask_rcnn_iter_1250000_cpu.pth'

CONFIG = dict(
    ROI_PER_IMG=32,
    HEAD_LR=2e-4,
    HEAD_WD=1e-3,
    REF_LR=5e-4,
    REF_WD=1e-3,
    RNN_LR=5e-4,
    RNN_WD=1e-3,
    BATCH_SIZE=32,
    EPOCH=5,
    ATT_DROPOUT_P=0.5,
    RANK_DROPOUT_P=0.5
)
LOG_INTERVAL = 50
VAL_INTERVAL = 1000


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


def compute_loss(predictor, criterion, device, enable_grad, roi_feats, roi_labels, word_feats, sent_len):
    with torch.autograd.set_grad_enabled(enable_grad):
        roi_feats = roi_feats.to(device)
        roi_labels = roi_labels.to(device)
        word_feats = word_feats.to(device)
        packed_sent_feats = pack_padded_sequence(word_feats, sent_len, enforce_sorted=False, batch_first=True)
        scores, *_ = predictor.forward(roi_feats, packed_sent_feats)
        loss = criterion(scores.flatten(), roi_labels.flatten())
    return loss


def main(args):
    tid = get_time_id()
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
    trn_dataset = DetBoxDataset(refdb, ctxdb, split='train', roi_per_img=CONFIG['ROI_PER_IMG'])
    val_dataset = DetBoxDataset(refdb, ctxdb, split='val', roi_per_img=CONFIG['ROI_PER_IMG'])
    trn_loader = DataLoader(trn_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=8)
    # Tensorboard writer
    tb_dir = 'tb/att_vanilla/{}'.format(tid)
    trn_wrt = SummaryWriter(os.path.join(tb_dir, 'train'))
    val_wrt = SummaryWriter(os.path.join(tb_dir, 'val'))
    # Build and init predictor
    predictor = AttVanillaPredictorV2(att_dropout_p=CONFIG['ATT_DROPOUT_P'],
                                      rank_dropout_p=CONFIG['RANK_DROPOUT_P'])
    init_att_vanilla_predictor(predictor)
    predictor.to(device)
    # Setup loss
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    # Setup optimizer
    ref_params = list(predictor.att_fc.parameters()) + list(predictor.rank_fc.parameters()) \
                 + list(predictor.vis_a_fc.parameters()) + list(predictor.vis_r_fc.parameters())
    ref_optimizer = optim.Adam(ref_params, lr=CONFIG['REF_LR'], weight_decay=CONFIG['REF_WD'])
    rnn_optimizer = optim.Adam(predictor.rnn.parameters(), lr=CONFIG['RNN_LR'], weight_decay=CONFIG['RNN_WD'])
    head_optimizer = optim.Adam(predictor.head.parameters(), lr=CONFIG['HEAD_LR'], weight_decay=CONFIG['HEAD_WD'])
    common_args = dict(mode='min', factor=0.5, verbose=True, threshold_mode='rel', patience=1)
    ref_scheduler = optim.lr_scheduler.ReduceLROnPlateau(ref_optimizer, min_lr=CONFIG['REF_LR']/100, **common_args)
    rnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(rnn_optimizer, min_lr=CONFIG['RNN_LR']/100, **common_args)
    head_scheduler = optim.lr_scheduler.ReduceLROnPlateau(head_optimizer, min_lr=CONFIG['HEAD_LR']/100, **common_args)
    # Start training
    step = 0
    trn_running_loss = 0.
    best_model = {'avg_val_loss': float('inf'), 'epoch': None, 'step': None, 'weights': None}
    tic = time()
    for epoch in range(CONFIG['EPOCH']):
        for trn_batch in trn_loader:
            # Train for one step
            step += 1
            predictor.train()
            loss = compute_loss(predictor, criterion, device, True, *trn_batch)
            head_optimizer.zero_grad()
            ref_optimizer.zero_grad()
            rnn_optimizer.zero_grad()
            loss.backward()
            head_optimizer.step()
            ref_optimizer.step()
            rnn_optimizer.step()
            trn_running_loss += loss.item()
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
                pbar = tqdm(total=len(val_dataset), ascii=True, desc='computing val loss')
                for val_batch in val_loader:
                    loss = compute_loss(predictor, criterion, device, False, *val_batch)
                    val_loss_list.append(loss.item())
                    pbar.update(val_batch[0].size(0))
                pbar.close()
                avg_val_loss = sum(val_loss_list) / len(val_loss_list)
                print('[VAL Loss] epoch {} step {}: {:.6f}'.format(epoch + 1, step, avg_val_loss))
                val_wrt.add_scalar('loss', avg_val_loss, step)
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
        save_path = 'output/att_vanilla_ckpt_{}_{}.pth'.format(tid, epoch + 1)
        torch.save(epoch_ckpt, save_path)
    # Save best model
    time_spent = int(time() - tic) // 60
    print('\nTraining completed in {} h {} m.'.format(time_spent // 60, time_spent % 60))
    print('Found model with lowest val loss at epoch {epoch} step {step}.'.format(**best_model))
    save_path = 'output/att_vanilla_{}_b.pth'.format(tid)
    torch.save(best_model['weights'], save_path)
    print('Saved best model weights to {}'.format(save_path))
    # Close summary writer
    trn_wrt.close()
    val_wrt.close()
    # Log training procedure
    model_info = {
        'type': 'att_vanilla_v2',
        'dataset': dataset_splitby,
        'config': CONFIG
    }
    with open('output/att_vanilla_{}.json'.format(tid), 'w') as f:
        json.dump(model_info, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='refcoco')
    parser.add_argument('--split-by', default='unc')
    parser.add_argument('--gpu-id', type=int, default=0)
    main(parser.parse_args())
