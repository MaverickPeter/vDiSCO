# Zhejiang University

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np
import torch
import tqdm
import pathlib
import wandb
import torch.nn as nn
from evaluate import GLEvaluator
from evaluate_overlap import OverlapEvaluator
from prnet.utils.params import TrainingParams, get_datetime
from prnet.models.loss import make_losses
from prnet.models.model_factory import model_factory
from prnet.datasets.dataset_utils import make_dataloaders
from prnet.utils.loss_utils import *
import time

VERBOSE = False
# USE_WANDB = True
USE_WANDB = False

def print_stats(stats, phase):
    if 'num_triplets' in stats:
        # For triplet loss
        s = '{} - Global loss: {:.6f}   Embedding norm: {:.4f}   Triplets (all/active): {:.1f}/{:.1f}'
        print(s.format(phase, stats['global_loss'], stats['avg_embedding_norm'], stats['num_triplets'],
                       stats['num_non_zero_triplets']))
    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}   Neg dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    if 'pos_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Pos loss: {:.4f}  Neg loss: {:.4f}'
        l += [stats['pos_loss'], stats['neg_loss']]

    if 'yaw_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Yaw loss: {:.6f} '
        l += [stats['yaw_loss']]

    if len(l) > 0:
        print(s.format(*l))


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def do_train(params: TrainingParams, resume=False, debug=False, visualize=False, weight=None, device='cpu'):
    # wandn_entity_name: Wights & Biases logging service entity name
    # Create model class

    s = get_datetime()
    model = model_factory(params.model_params)
    model_name = 'model_' + params.model_params.model + '_' + s
    print('Model name: {}'.format(model_name))
    weights_path = create_weights_folder()

    model_pathname = os.path.join(weights_path, model_name)
    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    # Move the model to the proper device before configuring the optimizer
    model.to(device)
    model = nn.DataParallel(model)
    corr2soft = Corr2Softmax(10., 0.)
    corr2soft = corr2soft.to(device)

    if resume:
        if 'netvlad' in params.model_params.model and 'pretrain' in params.model_params.model:
            checkpoint = torch.load(weight, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            checkpoint = torch.load(weight, map_location=device)
            model.load_state_dict(checkpoint, strict=True)

            # model.load_state_dict(checkpoint['model'], strict=True)
            # corr2soft.load_state_dict(checkpoint['corr2soft'], strict=True)


    # set up dataloaders
    dataloaders = make_dataloaders(params, debug=debug, device=device)

    print('Model device: {}'.format(device))

    gl_loss_fn, yaw_loss_fn = make_losses(params)

    # Training elements
    if params.weight_decay is None or params.weight_decay == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs+1,
                                                                   eta_min=params.min_lr)
        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.1)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(params.scheduler))

    radius = [2, 5, 10, 20]

    evaluator_test_set = GLEvaluator(params.dataset_folder, params.dataset, params.test_file, device=device,
                                            params=params.model_params, radius=radius, k=20, n_samples=None)

    ###########################################################################
    # Initialize Weights&Biases logging service
    ###########################################################################

    params_dict = {e: params.__dict__[e] for e in params.__dict__ if e != 'model_params'}
    model_params_dict = {"model_params." + e: params.model_params.__dict__[e] for e in params.model_params.__dict__}
    params_dict.update(model_params_dict)
    if USE_WANDB:
        wandb.init(project='Fusion_GL', config=params_dict)

    ###########################################################################
    #
    ###########################################################################

    phases = ['train', 'val']

    # Training statistics
    stats = {e: [] for e in phases}
    stats['eval'] = []

    for epoch in tqdm.tqdm(range(1, params.epochs + 1)):
        for phase in phases:
            if 'train' in phase:
                model.train()
            else:
                model.eval()

            running_stats = []  # running stats for the current epoch
            count_batches = 0

            if phase == 'train':
                global_phase = 'global_train'
            elif phase == 'val':
                global_phase = 'global_val'
            f = time.time()
            print("------ ", phase, "------ ")

            # !!! Below loop will skip some batches in the dataset having larger number of batches
            # for (batch, positives_mask, negatives_mask), local_batch in tqdm.tqdm(zip(dataloaders[global_phase], dataloaders[local_phase])):
            for batch, positives_mask, negatives_mask, yaws, gt_overlaps in tqdm.tqdm(dataloaders[global_phase]):
                # batch is (batch_size, n_points, 3) tensor
                # labels is list with indexes of elements forming a batch
                count_batches += 1
                batch_stats = {}

                if debug and count_batches > 2:
                    break

                # Move everything to the device
                for key in batch.keys():
                    if batch[key] is not None:
                        batch[key] = batch[key].to(device)

                n_positives = torch.sum(positives_mask).item()
                n_negatives = torch.sum(negatives_mask).item()

                if n_positives == 0 or n_negatives == 0:
                    # Skip a batch without positives or negatives
                    print('WARNING: Skipping batch without positive or negative examples')
                    continue

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # Compute embeddings of all elements
                    y = model(batch)

                    if params.use_overlap:
                        extracted_l = y['extracted_l']
                        extracted_r = y['extracted_r']
                        overlaps = y['overlap']
                    else:
                        embeddings = y['global']

                    yaw_loss = None
                    if 'spectrum' in y.keys():
                        spectrums = y['spectrum']
                        yaw_loss, yaw_stats = yaw_loss_fn(spectrums, yaws, corr2soft, positives_mask, negatives_mask)
                        batch_stats['yaw_loss'] = yaw_loss

                    if params.use_overlap:
                        loss, temp_stats = gl_loss_fn(overlaps, gt_overlaps)
                    else:
                        loss, temp_stats, _ = gl_loss_fn(embeddings, positives_mask, negatives_mask)

                    batch_stats['global_loss'] = loss.item()

                    temp_stats = tensors_to_numbers(temp_stats)
                    batch_stats.update(temp_stats)

                    batch_stats['loss'] = loss.item()

                    if yaw_loss is not None:
                        total_loss = loss
                    else:
                        total_loss = loss

                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

                running_stats.append(batch_stats)

            # ******* PHASE END *******
            # Compute mean stats for the epoch
            epoch_stats = {}
            for key in running_stats[0].keys():
                temp = [e[key] for e in running_stats]
                epoch_stats[key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(epoch_stats, phase)

        # ******* EPOCH END *******

        if scheduler is not None:
            scheduler.step()

        if epoch % params.save_freq == 0:
            torch.save({'model':model.state_dict(), 'corr2soft':corr2soft.state_dict()}, model_pathname + "_" + str(epoch) + ".pth")

        metrics = {'train': {}, 'val': {}, 'test': {}}
        metrics['train']['loss'] = stats['train'][-1]['loss']
        metrics['val']['loss'] = stats['val'][-1]['loss']

        if 'num_triplets' in stats['train'][-1]:
            metrics['train']['active_triplets'] = stats['train'][-1]['num_non_zero_triplets']
            metrics['val']['active_triplets'] = stats['val'][-1]['num_non_zero_triplets']
        elif 'num_pairs' in stats['train'][-1]:
            metrics['train']['active_pos'] = stats['train'][-1]['pos_pairs_above_threshold']
            metrics['train']['active_neg'] = stats['train'][-1]['neg_pairs_above_threshold']
            metrics['val']['active_pos'] = stats['val'][-1]['pos_pairs_above_threshold']
            metrics['val']['active_neg'] = stats['val'][-1]['neg_pairs_above_threshold']


        if epoch % params.eval_freq == 0:
            print("------ eval------ ")
            global_test_metrics = evaluator_test_set.evaluate(model)
            evaluator_test_set.print_results(global_test_metrics)

            metrics['test']['recall'] = global_test_metrics['recall']

        if USE_WANDB:
            wandb.log(metrics)

        if params.batch_expansion_th is not None:
            # Dynamic batch expansion
            epoch_train_stats = stats['train'][-1]
            if 'num_non_zero_triplets' not in epoch_train_stats:
                print('WARNING: Batch size expansion is enabled, but the loss function is not supported')
            else:
                # Ratio of non-zero triplets
                rnz = epoch_train_stats['num_non_zero_triplets'] / epoch_train_stats['num_triplets']
                if rnz < params.batch_expansion_th:
                    dataloaders['global_train'].batch_sampler.expand_batch()

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    print('')

    # Save final model weights
    final_model_path = model_pathname + '_final.pth'
    torch.save(model.state_dict(), final_model_path)

    # Evaluate the final model using all samples
    evaluator_test_set = GLEvaluator(params.dataset_folder, params.dataset, params.test_file, params=params.model_params,
                                            device=device, radius=radius, k=20, n_samples=None)
    nclt = evaluator_test_set.evaluate(model)

    print(params.dataset + '_eval:')
    evaluator_test_set.print_results(nclt)
    print('.')

    # Append key experimental metrics to experiment summary file
    # model_params_name = os.path.split(params.model_params.model_params_path)[1]
    # config_name = os.path.split(params.params_path)[1]
    # _, model_name = os.path.split(model_pathname)
    # prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)
    # export_eval_stats("experiment_results.txt", prefix, loc_stats)


def export_eval_stats(file_name, prefix, stats):
    s = prefix
    # Print results on the final model
    with open(file_name, "a") as f:
        train_loss = stats['train_stats']['train'][-1]['loss']
        train_nnz = stats['train_stats']['train'][-1]['num_non_zero_triplets']
        val_loss = stats['train_stats']['val'][-1]['loss']
        val_nnz = stats['train_stats']['val'][-1]['num_non_zero_triplets']
        recall_5 = stats['eval']['mulran_sejong_eval']['recall'][5][0]
        recall_20 = stats['eval']['mulran_sejong_eval']['recall'][20][0]
        s += ", {:0.3f}, {:0.1f}, {:0.3f}, {:0.1f}, {:0.3f}, {:0.3f}\n"
        s = s.format(train_loss, train_nnz, val_loss, val_nnz, recall_5, recall_20)
        f.write(s)

    metrics = {'recall_5': recall_5, 'recall_20': recall_20}
    if USE_WANDB:
        wandb.log(metrics)


def create_weights_folder():
    # Create a folder to save weights of trained models
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
    return weights_path
