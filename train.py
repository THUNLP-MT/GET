#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED

########### Import your packages below ##########
from data.dataset import BlockGeoAffDataset, PDBBindBenchmark, MixDatasetWrapper, DynamicBatchWrapper
from data.atom3d_dataset import LEPDataset, LBADataset
import models
import trainers
from utils.nn_utils import count_parameters
from data.pdb_utils import VOCAB


def parse():
    parser = argparse.ArgumentParser(description='training')
    # data
    parser.add_argument('--train_set', type=str, required=True, help='path to train set')
    parser.add_argument('--valid_set', type=str, default=None, help='path to valid set')
    parser.add_argument('--pdb_dir', type=str, default=None, help='directory to the complex pdbs (required if not preprocessed in advance)')
    parser.add_argument('--task', type=str, default=None,
                        choices=['PPA', 'PLA', 'LEP', 'AffMix', 'PDBBind', 'NL', 'EC'],
                        help='PPA: protein-protein affinity, ' + \
                             'PLA: protein-ligand affinity (small molecules), ' + \
                             'LEP: ligand efficacy prediction, ' + \
                             'PDBBind: pdbbind benchmark, ')
    parser.add_argument('--train_set2', type=str, default=None, help='path to another train set if task is PretrainMix')
    parser.add_argument('--valid_set2', type=str, default=None, help='path to another valid set if task is PretrainMix')
    parser.add_argument('--train_set3', type=str, default=None, help='path to the third train set (in NL task)')
    parser.add_argument('--fragment', type=str, default=None, choices=['PS_300', 'PS_500'], help='fragmentation on small molecules')

    # training related
    parser.add_argument('--pretrain', action='store_true', help='pretraining mode')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4, help='final learning rate')
    parser.add_argument('--warmup', type=int, default=0, help='linear learning rate warmup')
    parser.add_argument('--max_epoch', type=int, default=10, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save model and logs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--valid_batch_size', type=int, default=None, help='batch size of validation, default set to the same as training batch size')
    parser.add_argument('--max_n_vertex_per_gpu', type=int, default=None, help='if specified, ignore batch_size and form batch with dynamic size constrained by the total number of vertexes')
    parser.add_argument('--valid_max_n_vertex_per_gpu', type=int, default=None, help='form batch with dynamic size constrained by the total number of vertexes')
    parser.add_argument('--patience', type=int, default=-1, help='patience before early stopping')
    parser.add_argument('--save_topk', type=int, default=-1, help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=SEED)

    # trial: noised structure
    parser.add_argument('--noisy_sigma', type=float, default=0.0, help='noise scale to simulate docked structure with errors')

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    # model
    parser.add_argument('--model_type', type=str, required=True, choices=['GET', 'GETPool', 'SchNet', 'EGNN', 'DimeNet', 'TorchMD', 'Equiformer', 'GemNet', 'MACE', 'LEFTNet'], help='type of model to use')
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of residue/atom embedding')
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of hidden states')
    parser.add_argument('--n_channel', type=int, default=1, help='number of channels')
    parser.add_argument('--n_rbf', type=int, default=1, help='Dimension of RBF')
    parser.add_argument('--cutoff', type=float, default=7.0, help='Cutoff in RBF')
    parser.add_argument('--n_head', type=int, default=1, help='Number of heads in the multi-head attention')
    parser.add_argument('--k_neighbors', type=int, default=9, help='Number of neighbors in KNN graph')
    parser.add_argument('--radial_size', type=int, default=16, help='Radial size in GET')
    parser.add_argument('--radial_dist_cutoff', type=float, default=5, help='Distance cutoff in radial graph')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')

    parser.add_argument('--atom_level', action='store_true', help='train atom-level model (set each block to a single atom in GET)')
    parser.add_argument('--hierarchical', action='store_true', help='train hierarchical model (atom-block)')
    parser.add_argument('--no_block_embedding', action='store_true', help='do not add block embedding')

    # load pretrain
    parser.add_argument('--pretrain_ckpt', type=str, default=None, help='path of the pretrained ckpt to load')

    return parser.parse_args()


def create_dataset(task, path, path2=None, path3=None, fragment=None):
    if task == 'PLA':
        # dataset = Atom3DLBA(path)
        dataset = LBADataset(path, fragment=fragment)
        if path2 is not None:  # add protein dataset
            dataset2 = BlockGeoAffDataset(path2)
            dataset = MixDatasetWrapper(dataset, dataset2)
    elif task == 'LEP':
        dataset = LEPDataset(path, fragment=fragment)
    elif task == 'PPA':
        dataset = BlockGeoAffDataset(path)
        if path2 is not None:  # add small molecule dataset
            dataset2 = LBADataset(path2, fragment=fragment)
            dataset = MixDatasetWrapper(dataset, dataset2)
    elif task == 'AffMix':
        dataset1 = BlockGeoAffDataset(path)
        dataset2 = LBADataset(path2, fragment=fragment)
        dataset = MixDatasetWrapper(dataset1, dataset2)
    elif task == 'PDBBind':
        dataset = PDBBindBenchmark(path)
        if path2 is not None:  # add protein dataset
            dataset2 = BlockGeoAffDataset(path2)
            dataset = MixDatasetWrapper(dataset, dataset2)
    elif task == 'NL':
        datasets = [BlockGeoAffDataset(path)]
        if path2 is not None:
            datasets.append(BlockGeoAffDataset(path2))
        if path3 is not None:
            datasets.append(LBADataset(path3, fragment=fragment))
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = MixDatasetWrapper(*datasets)
    elif task == 'EC':
        dataset = ECDataset(path)
    else:
        raise NotImplementedError(f'Dataset for {task} not implemented!')
    return dataset


def create_trainer(model, train_loader, valid_loader, config):
    model_type = type(model)
    if model_type == models.AffinityPredictor or model_type == models.NoisedAffinityPredictor:
        Trainer = trainers.AffinityTrainer
    elif model_type == models.GraphPairClassifier:
        Trainer = trainers.GraphPairClassificationTrainer
    elif model_type == models.GraphClassifier:
        Trainer = trainers.GraphClassificationTrainer
    elif model_type == models.DenoisePretrainModel:
        Trainer = trainers.PretrainTrainer
    elif model_type == models.GraphMultiBinaryClassifier:
        Trainer = trainers.ECTrainer
    else:
        raise NotImplementedError(f'Trainer for model type {model_type} not implemented!')
    return Trainer(model, train_loader, valid_loader, config)


def main(args):
    setup_seed(args.seed)
    VOCAB.load_tokenizer(args.fragment)
    # torch.autograd.set_detect_anomaly(True)
    model = models.create_model(args)

    ########### load your train / valid set ###########
    train_set = create_dataset(args.task, args.train_set, args.train_set2, args.train_set3, args.fragment)
    if args.valid_set is not None:
        valid_set = create_dataset(args.task, args.valid_set, args.valid_set2, fragment=args.fragment)
        print_log(f'Train: {len(train_set)}, validation: {len(valid_set)}')
    else:
        valid_set = None
        print_log(f'Train: {len(train_set)}, no validation')
    if args.max_n_vertex_per_gpu is not None:
        if args.valid_max_n_vertex_per_gpu is None:
            args.valid_max_n_vertex_per_gpu = args.max_n_vertex_per_gpu
        train_set = DynamicBatchWrapper(train_set, args.max_n_vertex_per_gpu)
        if valid_set is not None:
            valid_set = DynamicBatchWrapper(valid_set, args.valid_max_n_vertex_per_gpu)
        args.batch_size, args.valid_batch_size = 1, 1
        args.num_workers = 1

    ########## set your collate_fn ##########
    collate_fn = train_set.collate_fn

    ########## define your model/trainer/trainconfig #########
    step_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    config = trainers.TrainConfig(args.save_dir, args.lr, args.max_epoch,
                                  warmup=args.warmup,
                                  patience=args.patience,
                                  grad_clip=args.grad_clip,
                                  save_topk=args.save_topk)
    config.add_parameter(step_per_epoch=step_per_epoch,
                         final_lr=args.final_lr)
    if args.valid_batch_size is None:
        args.valid_batch_size = args.batch_size

    if len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=args.shuffle)
        if args.max_n_vertex_per_gpu is None:
            args.batch_size = int(args.batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {args.batch_size}')
    else:
        args.local_rank = -1
        train_sampler = None

    if args.local_rank <= 0:
        if args.max_n_vertex_per_gpu is not None:
            print_log(f'Dynamic batch enabled. Max number of vertex per GPU: {args.max_n_vertex_per_gpu}')
        if args.pretrain_ckpt:
            print_log(f'Loaded pretrained checkpoint from {args.pretrain_ckpt}')
        print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=(args.shuffle and train_sampler is None),
                              sampler=train_sampler,
                              collate_fn=collate_fn)
    if valid_set is not None:
        valid_loader = DataLoader(valid_set, batch_size=args.valid_batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)
    else:
        valid_loader = None
    trainer = create_trainer(model, train_loader, valid_loader, config)
    trainer.set_valid_requires_grad('pretrain' in args.task.lower())
    trainer.train(args.gpus, args.local_rank)

    json.dump(vars(args), open(os.path.join(trainer.model_dir, 'namespace.json'), 'w'))
    
    return trainer.topk_ckpt_map


if __name__ == '__main__':
    args = parse()
    main(args)
