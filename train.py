import copy
import math
import time

from torch import nn, optim
from torch.optim import Adam
from compression import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets, transforms
from utils import prepare_data
from torch.nn import functional as F
from utils import epoch_time

import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import os
import numpy as np

from conf import *
from mingpt.model import GPT
from mingpt.simple_vit import SimpleViT

##################################################
from dgc import DGCCompressor                    #
dgc_compression_ratio = 0.00201                   #
compressor = DGCCompressor(dgc_compression_ratio)#                          
##################################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


from math import floor
def topk(x,sr):
    x_1d = x.view(-1)
    dev = torch.device(x.device)
    _, idx = torch.abs(x_1d).topk(floor(sr * len(x_1d)),sorted=False)
    out = torch.zeros(len(x_1d),device=dev)
    out[idx] = x_1d[idx]
    x_1d.scatter_(0,idx,0)
    return out.reshape(x.shape),x_1d


def dgc_compress(model, flatten_param):
    out = torch.zeros_like(flatten_param)
    start = 0
    for name, param in model.named_parameters():
        if "experts" in name.split('.') or "base_expert" in name.split('.'): continue
        compressed_grad, ctx = compressor.compress(param.grad, name)
        restored_grad = compressor.decompress(compressed_grad, ctx)
        bias = restored_grad.numel()
        out[start:start + bias] = restored_grad.flatten()
        start += bias
    return out

def randk(x,sr):
    x_1d = x.view(-1)
    dev = torch.device(x.device)
    idx = torch.randint(low=0,high=len(x_1d),size=(floor(sr*len(x_1d)),),device=dev)
    out = torch.zeros(len(x_1d),device=dev)
    out[idx] = x_1d[idx]*(1/sr)
    return out.reshape(x.shape)


def tg(x,scale):
    x_1d = x.view(-1)
    max_ = torch.max(torch.abs(x_1d))/scale
    p_i = torch.clamp(torch.abs(x_1d) / max_, min=0.00000001, max=0.9999999)
    idx = torch.where((torch.rand(len(x_1d), device = x.device) < p_i) == True)[0]
    value = torch.sign(x_1d[idx]) * max_
    out = torch.zeros(len(x_1d),device=x.device)
    out[idx] = value
    return out.reshape(x.shape),len(idx)/len(x_1d)

def train(model, iterator, optimizer, scheduler, device, args, epoch, last_residuals) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    all_loss = 0.
    a2a_time = 0
    prepare_time = 0
    dispatch_time = 0
    combine_time = 0
    backward_time = 0
    allreduce_time = 0
    log_interval = 100
    start_time = time.time()
    new_residuals = last_residuals

    local_gradients = {}
    local_gradients_shape = {}
    #non_moe = []
    for name, parameters in model.named_parameters():
        if "experts" in name.split('.') or "base_expert" in name.split('.'):
            local_gradients[name] = 0
            local_gradients_shape[name] = parameters.shape
    #    else:
    #        non_moe.append(parameters.view(-1))
    #residuals = torch.zeros_like(torch.cat([tensor.view(-1) for tensor in non_moe]))
    sp_list=[]
    interval = 1
    num_batches = len(iterator)
    for iter, (data, targets) in enumerate(iterator):
        if (iter / 1).is_integer():   # subset training
            s=time.time()
            if task =="wikitext2" or task == "wikitext103":
                '''gpt'''
                output = model(data.to(device), targets.to(device))
                loss = output[-1]
                if (iter / interval).is_integer():
                    optimizer.zero_grad()
                loss.backward()
            else:
                '''vit'''
                output = model(data.to(device))
                loss = nn.CrossEntropyLoss()(output, targets.to(device))
                if (iter / interval).is_integer():
                    optimizer.zero_grad()
                loss.backward()
            #print('training time',time.time()-s)
            #######################################
            if (iter / interval).is_integer():    # dense
            #if iter==0: # sparse                 #
            #if iter==-1: # no all-reduce         #
            #######################################
                compress=True
                if compress:
                    #print('compress',iter)
                    """####### 非moe层all reduce过程的压缩(分层+积累) ##########"""
                    ss = time.perf_counter()
                    t1 = time.time()
                    grads = []
                    for name, parameters in model.named_parameters():
                        if "experts" in name.split('.') or "base_expert" in name.split('.'): pass
                        else:
                            grads.append(parameters.grad.data.view(-1))
                    flattened_grads = torch.cat([tensor.view(-1) for tensor in grads])
                    lengths = [len(tensor) for tensor in grads]
                    ###################################################################################
                    # output, new_residuals = topk(flattened_grads + last_residuals,0.00201)       # top-k
                    #output=flattened_grads                                                       # no compress
                    # output,sp = tg(flattened_grads,10);sp_list.append(sp)                        # RS ours
                    output = dgc_compress(model, flattened_grads)                                # dgc
                    # output = randk(flattened_grads, 0.26)                                     # rand-k
                    ###################################################################################
                    restored_list = torch.split(output, lengths)
                    index=0
                    for name, parameters in model.named_parameters():
                        if "experts" in name.split('.') or "base_expert" in name.split('.'): pass
                        else:
                            parameters.grad.data = restored_list[index].reshape(parameters.shape)
                            dist.all_reduce(tensor=parameters.grad, op=dist.ReduceOp.SUM)
                            parameters.grad = parameters.grad / dist.get_world_size()
                            index+=1
                    allreduce_time += time.time() - t1
                    #print('compressing:{}'.format(time.perf_counter()-ss))
                    """##############################################"""
                else:
                    #print('no compress',iter)
                    t1 = time.time()
                    for name, parameters in model.named_parameters():
                        if "experts" in name.split('.') or "base_expert" in name.split('.'):
                            local_gradients[name] += torch.square(parameters.grad).data
                        else: # all_reduce: 各节点数据相加，并将结果广播给所有进程
                            """#####逐层topk无本地积累#####"""
                            # parameters.grad.data = tg(parameters.grad.data)
                            # parameters.grad.data = topk(parameters.grad.data, 0.1)
                            # parameters.grad.data = qsgd(parameters.grad.data,8)
                            """##########################"""
                            dist.all_reduce(tensor=parameters.grad.data, op=dist.ReduceOp.SUM)
                            parameters.grad = parameters.grad / dist.get_world_size()
                    allreduce_time += time.time() - t1

                #optimizer.zero_grad()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            #scheduler.step()

            total_loss += loss.item()
            all_loss += loss.item()

            if iter % log_interval == 0 and iter > 0:
                lr = optimizer.param_groups[0]['lr']
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(cur_loss)
                print(f'| epoch {epoch:3d} | {iter:5d}/{num_batches:5d} batches | '
                      f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
                      f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                #scheduler.step(ppl)
                total_loss = 0
                import sys
                sys.stdout.flush()
                start_time = time.time()
            else: pass
    return all_loss / num_batches, a2a_time, prepare_time, dispatch_time, combine_time, backward_time, allreduce_time, local_gradients, local_gradients_shape, new_residuals, np.mean(sp_list)

# task = "wikitext2"    # gpt-small(原gpt-mini)
task = "wikitext103"  # gpt-large
#task = "cifar10"      # gpt-small(原vit-simple) xxxxx
#task = "cifar100"     # vit-small(原vit-simple)
#task = "tinyimgnet"   # gpt-large

def evaluate_gtp(model, iterator, device) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i, (data, targets) in enumerate(iterator):
            output = model(data.to(device), targets.to(device))
            loss = output[1]
            total_loss += loss
    return total_loss / (len(iterator) - 1)

def evaluate_vit(model, val_loader, device):
    model.eval()
    test_correct = 0
    test_correct_3 = 0
    test_correct_5 = 0
    for step, batch in enumerate(val_loader):
        inputs, targets = batch
        images, labels = inputs.to(device), targets.to(device)
        outs = model.forward(images)
        acc = torch.eq(outs.argmax(-1), labels).float().mean().item()
        acc3 = torch.eq(outs.topk(3, dim=-1, sorted=True, largest=True).indices, labels.reshape(-1, 1)).float().mean().item() * 3
        acc5 = torch.eq(outs.topk(5, dim=-1, sorted=True, largest=True).indices, labels.reshape(-1, 1)).float().mean().item() * 5
        test_correct += acc
        test_correct_3 += acc3
        test_correct_5 += acc5
    acc = test_correct / len(val_loader)
    acc3 = test_correct_3 / len(val_loader)
    acc5 = test_correct_5 / len(val_loader)
    print("val acc: {}, val acc3: {}, val acc5: {}".format(acc, acc3, acc5))
    return acc, acc3, acc5


def run(best_loss, args):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    total_epoch = task_config[task]["epoch"]
    distributed_config = {
        'local_rank': args.local_rank,
        'global_rank': args.global_rank,
        'nproc_per_node': args.nproc_per_node,
        'nnode': args.nnode,
        'rank': args.local_rank + args.global_rank * args.nproc_per_node,
        'backend': args.backend,
        'master_addr': args.master_addr,
        'master_port': args.master_port,
    }
    train_config = {
        'train_batch_size': args.train_batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'gamma': args.gamma,
        'use_moe': args.use_moe,
        'log_interval': args.log_interval,
    }
    test_config = {
        'test_batch_size': args.test_batch_size,
    }

    device = "cuda:{}".format(args.local_rank) if torch.cuda.is_available() else "cpu"

    backend = distributed_config['backend']
    rank = distributed_config['rank']
    if backend == 'nccl':
        os.environ['NCCL_IB_DISABLE'] = '1'
    world_size = distributed_config['nnode'] * distributed_config['nproc_per_node']

    # 等待进程组创建，没有足够数量的进程加入会在此处阻塞
    dist.init_process_group(backend, init_method='tcp://{}:{}'.format(distributed_config['master_addr'],
                            distributed_config['master_port']), rank=rank, world_size=world_size)
    
    print("begin to prepare task: ", task)
    if task == "wikitext2" or task == "wikitext103":
        data_config = prepare_data(dataset_name=task, input_size=64, bptt=1)
        
        sampler = DistributedSampler(data_config['train_dataset'])
        
        train_loader = DataLoader(
            dataset=data_config['train_dataset'],
            batch_size=task_config[task]["batch_size"],
            sampler=sampler
        )

        val_loader = DataLoader(
            dataset=data_config['val_dataset'],
            batch_size=task_config[task]["batch_size"],
        )

        test_loader = DataLoader(
            dataset=data_config['test_dataset'],
            batch_size=task_config[task]["batch_size"],
        )

        model_config = GPT.get_default_config()
        model_config.model_type = task_config[task]["model"]    # gpt-mini
        model_config.vocab_size = len(data_config['vocab'])
        model_config.block_size = 1024
        model_config.experts_num = args.experts_num
        model_config.device = device
        model = GPT(model_config).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=task_config[task]["init_lr"]) # gpt
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, task_config[task]["init_lr"], gamma=task_config[task]["gamma"])
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.95, patience=patience, verbose=True)
    else:
        if task == "cifar10" or task == "cifar100":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.Resize(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821])
            ])

            transform_test = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821])
            ])
            
        if task == "cifar10":
            training_set = datasets.CIFAR10(root='/GPUFS/hit_wxia_1/ywh/code/data', train=True, download=False, transform=transform_train)
            test_set = datasets.CIFAR10(root='/GPUFS/hit_wxia_1/ywh/code/data', train=False, download=False, transform=transform_test)
        elif task == "cifar100":
            training_set = datasets.CIFAR100(root='/GPUFS/hit_wxia_1/ywh/code/data', train=True, download=False, transform=transform_train)
            test_set = datasets.CIFAR100(root='/GPUFS/hit_wxia_1/ywh/code/data', train=False, download=False, transform=transform_test)
        else:
            training_set = datasets.ImageFolder(root=os.path.join('/GPUFS/hit_wxia_1/ywh/code/data/tiny-imagenet-200', 'train'),
                                            transform=transform_train)
            test_set = datasets.ImageFolder(root=os.path.join('/GPUFS/hit_wxia_1/ywh/code/data/tiny-imagenet-200', 'val'),
                                transform=transform_test)
            

        sampler = DistributedSampler(training_set) if dist.is_available() else None
        train_loader = torch.utils.data.DataLoader(dataset=training_set,
                                                   #batch_size=256,   # original
                                                   batch_size=task_config[task]["batch_size"],
                                                   sampler=sampler,
                                                   shuffle=False,
                                                   pin_memory=False)
        val_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                 batch_size=task_config[task]["batch_size"],
                                                 shuffle=True,
                                                 pin_memory=False)

        # model = SimpleViT(image_size=32, patch_size=4, num_classes=num_classes, dim=512, depth=6, heads=8, mlp_dim=512,
                        #   dev=device, experts_num=args.experts_num).to(device)
        model = SimpleViT(image_size=32, patch_size=4, num_classes=task_config[task]["num_classes"], dim=task_config[task]["model_dim"], depth=task_config[task]["model_depth"], \
                          heads=task_config[task]["model_heads"], mlp_dim=task_config[task]["model_mlp_dim"],
                          dev=device, experts_num=[args.experts_num for i in range(world_size)]).to(device)

        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9,0.999), eps=1e-08)  # vit
        optimizer = torch.optim.Adam(model.parameters(), lr=task_config[task]["init_lr"], betas=task_config[task]["betas"], eps=task_config[task]["adam_eps"])
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, task_config[task]["init_lr"], gamma=task_config[task]["gamma"])
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # optimizer = Adam(params=model.parameters(),
    #                 lr=init_lr,
    #                 weight_decay=weight_decay,
    #                 eps=adam_eps)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
    #                                                 verbose=True,
    #                                                 factor=factor,
    #                                                 patience=patience)

    train_losses, val_losses, bleus, train_ppl, val_ppl = [], [], [], [], []
    val_losses3, val_losses5 = [], []
    total_time, a2a_time_list, prepare_time_list, dispatch_time_list, combine_time_list, allreduce_time_list, backward_time_list = [], [], [], [], [], [], []
    #################################################
    compressor.initialize(model.named_parameters())#
    #################################################
    non_moe = []
    for name, parameters in model.named_parameters():
        if "experts" in name.split('.') or "base_expert" in name.split('.'): pass
        else:
            non_moe.append(parameters.view(-1))
    residuals = torch.zeros_like(torch.cat([tensor.view(-1) for tensor in non_moe]))
    SP_list = []
    for step in range(total_epoch):
        sampler.set_epoch(step)
        #compressor.warmup_compress_ratio(step)
        start_time = time.perf_counter()
        train_loss, a2a_time, prepare_time, dispatch_time, combine_time, backward_time, allreduce_time, local_gradients, local_gradients_shape,residuals,SP = train(model, train_loader, optimizer, scheduler, device, args,step,residuals)
        SP_list.append(SP);print('本轮平均sp',SP)
        end_time = time.perf_counter()
        # with open("result_record/rank{}_record.txt".format(dist.get_rank()), "a+") as f:
        #     for layer_id, layer in enumerate(model.transformer.layers):
        #         f.write("-----------------------------------------------------------------\n")
        #         f.write("epoch{} layer{} experts throughput: {}\n".format(step, layer_id, layer.moe.experts_throughput.tolist()))
        #         f.write("epoch{} layer{} gpu throughput: {}\n".format(step, layer_id, layer.moe.gpu_throughput.tolist()))

        if task=="wikitext2" or task == "wikitext103": valid_loss = evaluate_gtp(model, val_loader, device)   # gpt
        else:       valid_loss, valid_loss3, valid_loss5 = evaluate_vit(model, val_loader, device)   # vit


        if step >= warmup:
            scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        try:
            val_losses3.append(valid_loss3)
            val_losses5.append(valid_loss5)
        except:
            pass        

        train_ppl.append(math.exp(train_loss))
        val_ppl.append(math.exp(valid_loss))
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(dist.get_rank()))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()


        f = open('result/val_loss.txt', 'w')
        f.write(str(val_losses))
        f.close()
        if task != "wikitext2" or task != "wikitext103":
            f = open('result/val_loss3.txt', 'w')
            f.write(str(val_losses3))
            f.close()
            f = open('result/val_loss5.txt', 'w')
            f.write(str(val_losses5))
            f.close()
        # if dist.get_rank() == 0:
            # print("a2a time: ", a2a_time * 2)
            # print("allreduce time: ", allreduce_time)
            # print("prepare time: ", prepare_time)
            # print("dispatch time: ", dispatch_time)
            # print("combine time: ", combine_time)
            # print("backward time: ", backward_time)
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s | \tTrain Loss: {train_loss:.3f} |'
              f' Train PPL: {math.exp(train_loss):7.3f} | Val Acc or Loss: {valid_loss:.3f} | Val PPL: {math.exp(valid_loss):7.3f}')
        # print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}') # gpt
        # print(f'Val ACC: {valid_loss:.3f}') # vit
        a2a_time_list.append(a2a_time)
        allreduce_time_list.append(allreduce_time)
        prepare_time_list.append(prepare_time)
        dispatch_time_list.append(dispatch_time)
        combine_time_list.append(combine_time)
        backward_time_list.append(backward_time)
        total_time.append(end_time - start_time)
    print('mean sp',np.mean(SP_list))
    train_losses_tensor = torch.tensor(train_losses, device=device)
    dist.all_reduce(tensor=train_losses_tensor, op=dist.ReduceOp.SUM)
    test_losses_tensor = torch.tensor(val_losses, device=device)
    dist.all_reduce(tensor=test_losses_tensor, op=dist.ReduceOp.SUM)
    # bleus_tensor = torch.tensor(bleus, device=device)
    # dist.all_reduce(tensor=bleus_tensor, op=dist.ReduceOp.SUM)
    train_ppl_tensor = torch.tensor(train_ppl, device=device)
    dist.all_reduce(tensor=train_ppl_tensor, op=dist.ReduceOp.SUM)
    test_ppl_tensor = torch.tensor(val_ppl, device=device)
    dist.all_reduce(tensor=test_ppl_tensor, op=dist.ReduceOp.SUM)
    total_time_tensor = torch.tensor(total_time, device=device)
    dist.all_reduce(tensor=total_time_tensor, op=dist.ReduceOp.SUM)
    if task != "wikitext2" or task != "wikitext103":
        val_acc3_tensor = torch.tensor(val_losses3, device=device)
        dist.all_reduce(tensor=val_acc3_tensor, op=dist.ReduceOp.SUM)
        val_acc5_tensor = torch.tensor(val_losses5, device=device)
        dist.all_reduce(tensor=val_acc5_tensor, op=dist.ReduceOp.SUM)
    if rank == 0:
        f = open('result_record.txt', 'a')
        f.write("GPU num: {0}\n".format(dist.get_world_size()))
        f.write("train loss list: {0}\n".format((train_losses_tensor/world_size).cpu().tolist()))
        f.write("test loss list: {0}\n".format((test_losses_tensor/world_size).cpu().tolist()))
        # f.write("bleu list: {0}\n".format((bleus_tensor/world_size).cpu().tolist()))
        f.write("train ppl list: {0}\n".format((train_ppl_tensor/world_size).cpu().tolist()))
        f.write("test ppl list: {0}\n".format((test_ppl_tensor/world_size).cpu().tolist()))
        # print("a2a time list: ", a2a_time_list)
        # print("allreduce time list: ", allreduce_time_list)
        # print("prepare time list: ", prepare_time_list)
        # print("dispatch time list: ", dispatch_time_list)
        # print("combine time list: ", combine_time_list)
        # print("backward time list: ", backward_time_list)
        if task != "wikitext2" or task != "wikitext103":
            f.write("test acc3 list: {0}\n".format((val_acc3_tensor/world_size).cpu().tolist()))
            f.write("test acc5 list: {0}\n".format((val_acc5_tensor/world_size).cpu().tolist()))
        f.write("total time list: {0}\n".format((total_time_tensor/world_size).cpu().tolist()))
        f.write("--------------------------------------------------\n")
        f.close()

    # f = open('final.txt', 'a')
    # # f.write(str(test_losses))
    # f.write("local mask number: {0}, mask ratio: {1}\n".format(args.experts_num, args.mask_ratio))
    # f.write("train loss list: {0}\n".format(train_losses))
    # f.write("test loss list: {0}\n".format(test_losses))
    # f.write("bleu list: {0}\n".format(bleus))
    # f.write("train ppl list: {0}\n".format(train_ppl))
    # f.write("test ppl list: {0}\n".format(val_ppl))
    # # print("a2a time list: ", a2a_time_list)
    # # print("allreduce time list: ", allreduce_time_list)
    # # print("prepare time list: ", prepare_time_list)
    # # print("dispatch time list: ", dispatch_time_list)
    # # print("combine time list: ", combine_time_list)
    # # print("backward time list: ", backward_time_list)
    # f.write("total time list: {0}\n".format(total_time))
    # f.write("--------------------------------------------------\n")
    # f.close()
    # print("train loss list: {0}".format(train_losses))
    # print("test loss list: {0}".format(test_losses))
    # print("bleu list: {0}".format(bleus))
    # print("train ppl list: {0}".format(train_ppl))
    # print("test ppl list: {0}".format(val_ppl))
    # # print("a2a time list: ", a2a_time_list)
    # # print("allreduce time list: ", allreduce_time_list)
    # # print("prepare time list: ", prepare_time_list)
    # # print("dispatch time list: ", dispatch_time_list)
    # # print("combine time list: ", combine_time_list)
    # # print("backward time list: ", backward_time_list)
    # print("total time list: ", total_time)


if __name__ == '__main__':
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--use_moe', action='store_true', default=True,
                        help='if disabling moe layer and using ffn layer instead')
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--global_rank", type=int, default=0)
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--backend", type=str, default='nccl')
    parser.add_argument("--master_addr", type=str, default='127.0.0.1')
    parser.add_argument("--master_port", type=int, default=12331)
    parser.add_argument("--nnode", type=int, default=1)
    parser.add_argument("--experts_num", type=int, default=5)
    args = parser.parse_args()

    # 开启多进程训练
    mp.set_start_method("spawn")
    p = mp.Process(target=run, args=(inf, args))
    p.start()
    p.join()







'''cifar10, tmgnet'''
# training_set = datasets.CIFAR10(root="../../data", train=True, download=True, transform=transforms.Compose(
#                             [transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]))
# test_set = datasets.CIFAR10(root="../../data", train=False, download=True, transform=transforms.Compose(
#                                     [ transforms.RandomCrop(32, padding=4),transforms.ToTensor(),
#                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
# training_set = datasets.ImageFolder(root=os.path.join('/home/wdl/project/data/tiny-imagenet-200', 'train'),
#                                     transform=transforms.Compose([transforms.RandomResizedCrop(224),
#                                     transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(
#                                       [0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821])]))
# test_set = datasets.ImageFolder(root=os.path.join('/home/wdl/project/data/tiny-imagenet-200', 'val'),
#                                 transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
#                              transforms.ToTensor(), transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821])]))


'''tiny image net'''
# num_classes=200
# transform_train = transforms.Compose(
#     [transforms.RandomCrop(32, padding=4),
#      transforms.Resize(32),
#      transforms.RandomHorizontalFlip(),
#      transforms.ToTensor(),
#      transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821])])
#
# transform_test = transforms.Compose(
#     [transforms.Resize(32),
#      transforms.ToTensor(),
#      transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821])])
# training_set = datasets.ImageFolder(root=os.path.join('/home/wdl/project/data/tiny-imagenet-200', 'train'),
#                                     transform=transform_train)
# test_set = datasets.ImageFolder(root=os.path.join('/home/wdl/project/data/tiny-imagenet-200', 'val'),
#                                 transform=transform_test)
