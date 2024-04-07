"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# GPU device setting
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# global distributed experts layout
experts_layout = {
    '10.250.248.109': [2, 2],
    '10.249.141.41': [2, 2],
}


task_config = {
    "wikitext2":{
        "batch_size": 35,
        "model": "gpt-mini",
        "init_lr": 1.0,
        "gamma": 0.95,
        "epoch": 10
    },
    "wikitext103":{
        "batch_size": 48,     # 18
        "model": "gpt-mini",  # gpt1-m
        "init_lr": 1.0,
        "gamma": 0.95,
        "epoch": 8            # 5
    },
    "cifar10":{
        "batch_size": 256,
        "num_classes": 10,
        "model_dim": 512,
        "model_depth": 6,
        "model_heads": 8,
        "model_mlp_dim": 512,
        "init_lr": 0.0001,
        "betas": (0.9, 0.999),
        "adam_eps": 1e-08,
        "gamma": 0.95,
        "epoch": 50
    },
    "cifar100":{
        "batch_size": 256,
        "num_classes": 100,
        "model_dim": 512,
        "model_depth": 6,
        "model_heads": 8,
        "model_mlp_dim": 512,
        "init_lr": 0.0001,
        "betas": (0.9, 0.999),
        "adam_eps": 1e-08,
        "gamma": 0.95,
        "epoch": 35
    },
    "tinyimgnet":{
        "batch_size": 18,
        "num_classes": 200,
        "model_dim": 640,
        "model_depth": 12,
        "model_heads": 12,
        "model_mlp_dim": 2560,
        "init_lr": 0.0001,
        "betas": (0.9, 0.999),
        "adam_eps": 1e-08,
        "gamma": 0.95,
        "epoch": 15
    }

}




# model parameter setting
batch_size = 24
max_len = 512
d_model = 64
n_layers = 1
n_heads = 8
ffn_hidden = 2048
# drop_prob = 0.1

# optimizer parameter setting
#init_lr = 1
#factor = 0.95
#adam_eps = 5e-9
#patience = 10
warmup = 0
#epoch = 5    # wiki2 20 | wiki103 10 | cifar100 50 | cifar10 100
clip = 1.0
weight_decay = 5e-6
inf = float('inf')
