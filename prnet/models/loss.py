import numpy as np
import torch
from pytorch_metric_learning import losses, miners, reducers
from pytorch_metric_learning.distances import LpDistance
from ..utils.params import TrainingParams
from ..utils.data_utils.poses import apply_transform
from prnet.utils.loss_utils import *

def make_losses(params: TrainingParams):
    if params.loss == 'BatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        gl_loss_fn = BatchHardTripletLossWithMasks(params.margin)
    elif params.loss == 'BatchHardContrastiveLoss':
        gl_loss_fn = BatchHardContrastiveLossWithMasks(params.pos_margin, params.neg_margin)
    elif params.loss == 'Overlap':
        gl_loss_fn = OverlapLoss(params)
    else:
        print('Unknown loss: {}'.format(params.loss))
        raise NotImplementedError

    yaw_loss_fn = YawLossWithMasks(params)

    return gl_loss_fn, yaw_loss_fn


class OverlapLoss:
    def __init__(self, params):
        self.params = params

    def __call__(self, overlaps, sample_truth):
        diff_value = torch.abs(overlaps.squeeze(-1) - sample_truth.cuda())
        sigmoidx = (diff_value[:,0] + 0.25) * 24 - 12
        loss = torch.mean(1 / (1 + torch.exp(-sigmoidx)))
        stats = {'yaw_loss': loss.item()}

        return loss, stats


class YawLossWithMasks:
    def __init__(self, params):
        self.celoss = torch.nn.CrossEntropyLoss(reduction="sum")
        self.params = params

    def __call__(self, spectrums, yaws, corr2soft, positives_mask, negatives_mask):
        dummy_labels = torch.arange(spectrums.shape[0])
        
        loss = torch.zeros(1).cuda()

        for ndx, mask in enumerate(positives_mask):
            qyaw = yaws[ndx].unsqueeze(0)
            qspecs = spectrums[ndx]

            pos_idx = dummy_labels[positives_mask[ndx]][0].item()
            pos_yaw = yaws[pos_idx].unsqueeze(0)
            pos_specs = spectrums[pos_idx]
            angle, corr = phase_corr(qspecs, pos_specs)
            corr = corr2soft(corr)

            yaw_diff = (qyaw-pos_yaw)
            if yaw_diff > torch.pi:
                yaw_diff -= torch.pi
            elif yaw_diff < -torch.pi:
                yaw_diff += torch.pi

            gt_yaw = torch.ceil(yaw_diff / torch.pi * self.params.model_params.theta / 2.) - 1.
            gt_yaw += (self.params.model_params.theta / 2)
            gt_yaw = gt_yaw.cuda()

            yaw_loss = self.celoss(corr.unsqueeze(0), gt_yaw.long())
            loss += yaw_loss.to(loss.device)

        stats = {'yaw_loss': loss.item()}
        loss = loss.item()

        return loss, stats


class HardTripletMinerWithMasks:
    # Hard triplet miner
    def __init__(self, distance):
        self.distance = distance
        # Stats
        self.max_pos_pair_dist = None
        self.max_neg_pair_dist = None
        self.mean_pos_pair_dist = None
        self.mean_neg_pair_dist = None
        self.min_pos_pair_dist = None
        self.min_neg_pair_dist = None

    def __call__(self, embeddings, positives_mask, negatives_mask):
        assert embeddings.dim() == 2
        d_embeddings = embeddings.detach()
        with torch.no_grad():
            hard_triplets = self.mine(d_embeddings, positives_mask, negatives_mask)
        return hard_triplets

    def mine(self, embeddings, positives_mask, negatives_mask):
        # Based on pytorch-metric-learning implementation
        dist_mat = self.distance(embeddings)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        self.max_pos_pair_dist = torch.max(hardest_positive_dist).item()
        self.max_neg_pair_dist = torch.max(hardest_negative_dist).item()
        self.mean_pos_pair_dist = torch.mean(hardest_positive_dist).item()
        self.mean_neg_pair_dist = torch.mean(hardest_negative_dist).item()
        self.min_pos_pair_dist = torch.min(hardest_positive_dist).item()
        self.min_neg_pair_dist = torch.min(hardest_negative_dist).item()
        return a, p, n


def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows


class BatchHardTripletLossWithMasks:
    def __init__(self, margin):
        self.margin = margin
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        # We use triplet loss with Euclidean distance
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=True, distance=self.distance,
                                                reducer=reducer_fn, collect_stats=True)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)

        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': self.loss_fn.reducer.triplets_past_filter,
                 'num_triplets': len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }

        return loss, stats, hard_triplets


class BatchHardContrastiveLossWithMasks:
    def __init__(self, pos_margin, neg_margin):
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        # We use contrastive loss with squared Euclidean distance
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.ContrastiveLoss(pos_margin=self.pos_margin, neg_margin=self.neg_margin,
                                              distance=self.distance, reducer=reducer_fn, collect_stats=True)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'pos_pairs_above_threshold': self.loss_fn.reducer.reducers['pos_loss'].pos_pairs_above_threshold,
                 'neg_pairs_above_threshold': self.loss_fn.reducer.reducers['neg_loss'].neg_pairs_above_threshold,
                 'pos_loss': self.loss_fn.reducer.reducers['pos_loss'].pos_loss.item(),
                 'neg_loss': self.loss_fn.reducer.reducers['neg_loss'].neg_loss.item(),
                 'num_pairs': 2*len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }

        return loss, stats, hard_triplets
