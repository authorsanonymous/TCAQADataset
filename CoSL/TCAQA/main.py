import os
import sys

sys.path.append('../')

import torch
import torch.nn as nn

from utils import *
from opts import *
from scipy import stats
from tqdm import tqdm
from dataset import VideoDataset
from models.i3d import InceptionI3d
from models.evaluator import Evaluator
from config import get_parser

def con_loss(feature,binary_scores):
    l_self = torch.einsum('nc,ck->nk', [feature, feature.T])


    positive_labels = torch.tensor([index for index, score in enumerate(binary_scores) if score == 1]).cuda()
    negative_labels = torch.tensor([index for index, score in enumerate(binary_scores) if score == 0]).cuda()

   
    pos_combinations = torch.cartesian_prod(positive_labels, positive_labels)
    pos_combinations = pos_combinations[pos_combinations[:, 0] != pos_combinations[:, 1]]

    
    neg_combinations = torch.cartesian_prod(negative_labels, negative_labels)
    neg_combinations = neg_combinations[neg_combinations[:, 0] != neg_combinations[:, 1]]

  
    l_pos_self = torch.log(torch.softmax(l_self, dim=-1))
    l_pos_self = l_pos_self / 0.07


    pos_values = l_pos_self[pos_combinations[:, 0], pos_combinations[:, 1]]
    neg_values = l_pos_self[neg_combinations[:, 0], neg_combinations[:, 1]]
    combined_values = torch.cat((pos_values, neg_values), dim=0)


    cross_combinations = torch.tensor([], dtype=torch.long) 
    reverse_combinations = torch.tensor([], dtype=torch.long)


    cross_combinations = torch.cartesian_prod(positive_labels, negative_labels)
    reverse_combinations = torch.cartesian_prod(negative_labels, positive_labels)
    all_combinations = torch.cat((cross_combinations, reverse_combinations),dim=0)


    l_neg_self = torch.log(1 - torch.softmax(l_self, dim=-1))
    l_neg_self = l_neg_self / 0.07
    
    
    cross_values = l_neg_self[all_combinations[:, 0], all_combinations[:, 1]]
    
    
    cl_pos_loss = -torch.sum(combined_values)
    cl_pos_loss /= combined_values.size(0)
    
    cl_neg_loss = torch.sum(cross_values)
    cl_neg_loss /= cross_values.size(0)



    total_loss = (cl_pos_loss + cl_neg_loss).mean()
    return total_loss


def get_models(args):
    """
    Get the i3d backbone and the evaluator with parameters moved to GPU.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    i3d = InceptionI3d().cuda()
    i3d.load_state_dict(torch.load(i3d_pretrained_path))

    if args.type == 'USDL-CoSL':
        evaluator = Evaluator(output_dim=output_dim['USDL-CoSL'], model_type='USDL-CoSL').cuda()
    else:
        evaluator = Evaluator(output_dim=output_dim['MUSDL-CoSL'], model_type='MUSDL-CoSL', num_judges=num_judges).cuda()

    if len(args.gpu.split(',')) > 1:
        i3d = nn.DataParallel(i3d)
        evaluator = nn.DataParallel(evaluator)
    return i3d, evaluator


def compute_score(model_type, probs, data):
    if model_type == 'USDL-CoSL':
        pred = probs.argmax(dim=-1) * (label_max / (output_dim['USDL-CoSL']-1))
    else:
        # calculate expectation & denormalize & sort
        judge_scores_pred = torch.stack([prob.argmax(dim=-1) * judge_max / (output_dim['MUSDL-CoSL']-1)
                                         for prob in probs], dim=1).sort()[0]  # N, 7

        # keep the median 3 scores to get final score according to the rule of diving
        pred = torch.sum(judge_scores_pred[:, 2:5], dim=1) * data['difficulty'].cuda()
    return pred


def compute_loss(model_type, criterion, probs, data):
    if model_type == 'USDL-CoSL':
        loss = criterion(torch.log(probs), data['soft_label'].cuda())
    else:
        loss = sum([criterion(torch.log(probs[i]), data['soft_judge_scores'][:, i].cuda()) for i in range(num_judges)])
    return loss


def get_dataloaders(args):
    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(VideoDataset('train', args),
                                                       batch_size=args.train_batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       worker_init_fn=worker_init_fn)

    dataloaders['test'] = torch.utils.data.DataLoader(VideoDataset('test', args),
                                                      batch_size=args.test_batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn)
    return dataloaders


def main(dataloaders, i3d, evaluator, base_logger, args):
    # print configuration
    print('=' * 40)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 40)

    criterion = nn.KLDivLoss()
    optimizer = torch.optim.Adam([*i3d.parameters()] + [*evaluator.parameters()],
                                 lr=args.lr, weight_decay=args.weight_decay)

    epoch_best = 0
    rho_best = 0
    for epoch in range(args.num_epochs):
        log_and_print(base_logger, f'Epoch: {epoch}  Current Best: {rho_best} at epoch {epoch_best}')

        for split in ['train', 'test']:
            true_scores = []
            pred_scores = []

            if split == 'train':
                i3d.train()
                evaluator.train()
                torch.set_grad_enabled(True)
            else:
                i3d.eval()
                evaluator.eval()
                torch.set_grad_enabled(False)

            for data in tqdm(dataloaders[split]):
                true_scores.extend(data['final_score'].numpy())
                videos = data['video'].cuda()
                videos.transpose_(1, 2)  # N, C, T, H, W

                batch_size, C, frames, H, W = videos.shape
                clip_feats = torch.empty(batch_size, 10, feature_dim).cuda()
                for i in range(9):
                    clip_feats[:, i] = i3d(videos[:, :, 10 * i:10 * i + 16, :, :]).squeeze(2)
                clip_feats[:, 9] = i3d(videos[:, :, -16:, :, :]).squeeze(2)

                binary_scores = []
                for i in range(0, len(data['final_score'])):
                    if data['final_score'][i] >= 60:
                        binary_score = 1
                    else:
                        binary_score = 0

                    binary_scores.append(binary_score)

                probs = evaluator(clip_feats.mean(1))
                preds = compute_score(args.type, probs, data)
                pred_scores.extend([i.item() for i in preds])

                if split == 'train':
                    con_loss = lbcl_loss(clip_feats.mean(1), binary_scores)
                    klloss = compute_loss(args.type, criterion, probs, data)
                    loss = klloss + 0.00001 * con_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            rho, p = stats.spearmanr(pred_scores, true_scores)

            log_and_print(base_logger, f'{split} correlation: {rho}')

        if rho > rho_best:
            rho_best = rho
            epoch_best = epoch
            log_and_print(base_logger, '-----New best found!-----')
            log_and_print(base_logger, f'*******************pred_scores：{pred_scores}')
            log_and_print(base_logger, f'*******************true_scores：{true_scores}')

            path = 'ckpts/' + str(rho) + '.pt'
            torch.save({'epoch': epoch,
                        'i3d': i3d.state_dict(),
                        'evaluator': evaluator.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'rho_best': rho_best}, path)


if __name__ == '__main__':

    args = get_parser().parse_args()

    if not os.path.exists('./exp'):
        os.mkdir('./exp')
    if not os.path.exists('./ckpts'):
        os.mkdir('./ckpts')

    init_seed(args)

    base_logger = get_logger(f'exp/{args.type}.log', args.log_info)
    i3d, evaluator = get_models(args)
    dataloaders = get_dataloaders(args)

    main(dataloaders, i3d, evaluator, base_logger, args)
