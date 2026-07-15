import argparse
import os
import shutil
import sys

from mpi4py import MPI
import pandas as pd
import yaml
import torch

from node_cnn1d import ClientCNN1D, ServerCNN1D
from node import Node

def init_node(rank: int, server_opt: str, server_lr: float, tau: float, beta: float) -> Node:
    available_optimizers = ['fedavg', 'fedavgm', 'fedadagrad', 'fedadam', 'fedyogi']
    if server_opt not in available_optimizers:
        raise ValueError(f'Server optimizer {server_opt} unavailable, must be in {available_optimizers}.')
    return ServerCNN1D(server_opt, server_lr, tau, beta) if rank == 0 else ClientCNN1D(rank)

def share_public_keys(node: Node) -> None:
    key_rank_pairs = comm.gather((node.rank, node.public_key), root=0)
    if node.rank == 0:
        key_rank_pairs = {r: cpk for r, cpk in key_rank_pairs}
        key_rank_pairs.pop(0)
        node.clients_public_keys = key_rank_pairs
    public_key = comm.bcast(node.public_key, root=0)
    if node.rank != 0:
        node.server_public_key = public_key

def share_symmetric_key(node: Node) -> None:
    if node.rank == 0:
        node.generate_symmetric_key()
        sk = [None] + node.get_symmetric_key()
    else:
        sk = None
    sk = comm.scatter(sk, root=0)
    if node.rank != 0:
        node.symmetric_key = sk

def initial_broadcast(node: Node, pretrained_weights: str, data: str, cfg: str, hyp: str, imgsz: int) -> None:
    if node.rank == 0:
        node.initialize_model(pretrained_weights)
        encrypted_data = [None] + node.get_weights(metadata=True)
        # Removed post_init_update because it is a YOLO-specific function
    else:
        encrypted_data = None
    encrypted_data = comm.scatter(encrypted_data, root=0)
    if node.rank != 0:
        node.set_weights(encrypted_data, metadata=True)

def federated_loop(node: Node, nrounds: int, epochs: int, saving_path: str, architecture: str, pretrained_weights: str,
                   data: str, bsz_train: int, bsz_val: int, imgsz: int, conf_thres: float, iou_thres: float, cfg: str,
                   hyp: str, workers: int) -> None:
    for kround in range(nrounds):
        share_symmetric_key(node)
        if kround == 0:
            initial_broadcast(node, pretrained_weights, data, cfg, hyp, imgsz)
        if node.rank != 0:
            node.train(nrounds, kround, epochs, architecture, data, bsz_train, imgsz, cfg, hyp, workers, saving_path)
            sd_encrypted = node.get_update()
        else:
            sd_encrypted = None
        sd_encrypted = comm.gather(sd_encrypted, root=0)
        if node.rank == 0:
            sd_encrypted.pop(0)
            node.aggregate(sd_encrypted)
            node.reparameterize(architecture)
            node.test(kround, saving_path, data, bsz_val, imgsz, conf_thres, iou_thres)
            sd_encrypted = [None] + node.get_weights(metadata=False)
        else:
            sd_encrypted = None
        sd_encrypted = comm.scatter(sd_encrypted, root=0)
        if node.rank != 0:
            node.set_weights(sd_encrypted, metadata=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrounds', type=int, default=30, help='number of communication rounds')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs executed per communication round')
    parser.add_argument('--server-opt', type=str, default='fedavg', help='aggregation algorithm/server-side optimizer')
    parser.add_argument('--server-lr', type=float, default=1., help='server learning rate')
    parser.add_argument('--tau', type=float, default=1e-3, help='server adaptivity for FedAdagrad, FedAdam and FedYogi')
    parser.add_argument('--beta', type=float, default=0.1, help='server momentum with FedAvgM')
    # Sửa mặc định của đường dẫn data sang C:\FederatedLearning\FL\core\data iov
    parser.add_argument('--data', type=str, default=r'C:\FederatedLearning\FL\core\data iov', help='data directory')
    parser.add_argument('--bsz-train', type=int, default=32, help='batch size used for training')
    parser.add_argument('--bsz-val', type=int, default=32, help='batch size used for evaluation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'resume'], help='execution mode')
    parser.add_argument('--resume-weights', type=str, default='', help='path to weights for resume or test mode')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Init custom CNN1D Node
    node = init_node(rank, args.server_opt, args.server_lr, args.tau, args.beta)
    node.get_device_info()

    # Tính toán nsamples để thuật toán FedAvg có trọng số biết được 
    if node.rank != 0:
        data_path = os.path.join(args.data, 'federated_data', f'client_{node.rank-1}.pt')
        data_dict = torch.load(data_path, map_location='cpu')
        node.nsamples = len(data_dict['y'])

    share_public_keys(node)

    saving_path = 'experiments_cnn1d'
    if node.rank == 0:
        os.makedirs(saving_path, exist_ok=True)
        os.makedirs(saving_path + '/weights/', exist_ok=True)
        os.makedirs(saving_path + '/run/', exist_ok=True)
    comm.Barrier()

    if args.mode == 'test':
        if node.rank == 0:
            if not args.resume_weights:
                raise ValueError("Must provide --resume-weights for test mode")
            node.initialize_model(args.resume_weights)
            node.test(kround=0, saving_path=saving_path, data=args.data, bsz=args.bsz_val, imgsz=0, conf=0.0, iou=0.0)
        comm.Barrier()
        sys.exit(0)

    federated_loop(
        node=node,
        nrounds=args.nrounds,
        epochs=args.epochs,
        saving_path=saving_path,
        architecture='cnn1d',
        pretrained_weights=args.resume_weights if args.mode == 'resume' else None,
        data=args.data,
        bsz_train=args.bsz_train,
        bsz_val=args.bsz_val,
        imgsz=0,
        conf_thres=0.0,
        iou_thres=0.0,
        cfg='',
        hyp='',
        workers=0
    )
