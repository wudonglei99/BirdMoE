

import torch.nn as nn
import torch.distributed as dist
import time
import math
from compression import *

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )
    def forward(self, x):
        return self.net(x)

# 定义gate，使用topk
class TopKGate(nn.Module):
    def __init__(self, input_size, num_experts, k, device='cpu'):
        super().__init__()
        self._input_size = input_size
        self.num_experts = num_experts
        self.k = k
        assert k >= 1 and k <= num_experts, \
        "k should satisfy: k >= 1 and k <= total experts number"
        self.device = device

        self.gate_layer = nn.Linear(input_size, num_experts, bias=True, device=device)
        self.smooth_func = nn.Softmax(dim=1)
        self.count = 0

    def gating(self, x):
        gate_res = self.gate_layer(x)
        topk_gate, topk_index = gate_res.topk(self.k, dim=1)
        return self.smooth_func(topk_gate), topk_index

    def forward(self, x):
        return self.gating(x)


# 定义topk压缩器
class TopKCompressor():
    def __init__(self, sparse_ratio, device='cpu') -> None:
        self.sparse_ratio = sparse_ratio
        self.device = device

    def compress(self, tensor, ctx=[]):
        shape = tensor.shape
        dtype = tensor.dtype
        ctx = shape, dtype
        return tensor, ctx

    def decompress(self, tensor, ctx=[]):
        shape, dtype = ctx
        decompressed_tensor = torch.zeros(shape, device=self.device,dtype=dtype)
        values, indices = tensor
        decompressed_tensor[indices].data = values.data
        return decompressed_tensor

# 无压缩器
class NoneCompressor():
    def __init__(self):
        pass

    def compress(self, tensor, ctx=[]):
        return tensor

    def decompress(self, tensor, ctx=[]):
        return tensor


from math import floor
def topk(x,sr):
    x_1d = x.view(-1)
    _, idx = torch.abs(x_1d).topk(floor(sr * len(x_1d)))
    out = torch.zeros(len(x_1d),device=x.device)
    out[idx] = x_1d[idx]
    # x_1d.scatter_(0,idx,0)
    return out.reshape(x.shape)#,x_1d

level = 6
class Stitcher(Function):
    @staticmethod
    def forward(ctx, a2a_input_tensor, a2a_output_tensor, b2e_split, e2b_split):
        """
        对于第一个all2all过程，a2a_input_tensor代表batch划分的数据，input_split是发往每个expert的样本个数。
            a2a_output_tensor代表通信交换完成的tensor，对应当前进程expert需要的输入。

        对于第二个all2all过程，a2a_input_tensor代表expert划分的数据，output_split是发往每个worker的样本个数。
            a2a_output_tensor代表通信交换完成的tensor，对应的是当前worker需要的输入(后面求loss等)。

        该类是建立all2all通信前后tensor的联系，使得反向传播的时候计算图可以连接起来。
        *dispatch和combine要想成一个环形，它们的e2b和b2e刚好反过来
        *dispatch的b2e是站在e的视角，combine的b2e是站在b的视角
        *dispatch的e2b是站在b的视角，combine的e2g是站在e的视角
        """
        ctx.save_for_backward(b2e_split, e2b_split)
        # print('foward b2e',b2e_split)  # dispatch：我分别从不同节点那里拿到了多少分数据。combine:我手头的数据划分给了哪些专家（总和固定）
        # print('foward e2b',e2b_split)  # dispatch的e2b是构造还回来数据所需的容器（总和固定）。combine:：expert手头的数据要原路返还给哪些节点
        return a2a_output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        b2e_split, e2b_split = ctx.saved_tensors

        b2e_split = b2e_split.tolist()
        e2b_split = e2b_split.tolist()

        input_grad_list = list(torch.split(grad_output, b2e_split, dim=0))

        """###### dispatch 和 combine反向传播all to all过程的压缩 ########## 0.0048s x n_layer"""
        for i, grad in enumerate(input_grad_list):
           if grad.numel() != 0:
              grad=comb_comp.apply(grad,level)
           input_grad_list[i]=grad
        """#################################################################################"""

        total_sample = sum(e2b_split)
        output_grad_list = list(torch.split(torch.zeros((total_sample, grad_output.shape[1]), device=grad_output.device), e2b_split, dim=0))
        handle = dist.all_to_all(output_tensor_list=output_grad_list, input_tensor_list=input_grad_list, async_op=True)
        handle.wait()

        temp_grad = torch.cat(output_grad_list)
        return temp_grad, None, None, None


# all2all转换器，连接在gate和experts之间
class All2AllManager():
    """
    Args:
        expert_id: the selected experts id, with size (batch_size, selected_experts_number)
        experts_num: the total number of experts or the list of the global experts layout
        gate_weight: the gate weight for each experts output
        device: the running device
    """
    def __init__(self, experts_layout, device='cpu'):
        self.rank = dist.get_rank()
        self.device = device

        """
        如果experts_layout是按照列表形式给出,如experts_layout = [2, 3, 4, 2],则代表四个卡参加,每个卡上面有2,3,4,2个experts
        如果是一个数字的话,则有这么多卡参加训练,默认一个卡上面一个expert
        """
        if type(experts_layout) == list:
            self.global_experts_num = sum(experts_layout)
            self.gpu_num = len(experts_layout)
            self.local_experts_num = experts_layout[self.rank]
            self.experts_layout = experts_layout
        else:
            self.global_experts_num = experts_layout
            self.gpu_num = experts_layout
            self.local_experts_num = 1
            self.experts_layout = [1] * experts_layout

        self.experts_traffic = torch.zeros(self.global_experts_num, device=self.device, dtype=torch.long)
        self.gpu_traffic = torch.zeros(self.gpu_num, device=self.device, dtype=torch.long)

        # 在第一次dispatch之前需要先知道每个expert会处理多少token
        self.num4experts_from_gpu_list = list(torch.zeros(self.local_experts_num * self.gpu_num, device=self.device, dtype=torch.long).chunk(self.gpu_num))



    def connect(self, expert_id, gate_weight=None):

        self.expert_id = expert_id
        self.batch_size, self.select_experts_num = expert_id.shape

        if gate_weight == None:
            self.origin_gate_weight = torch.ones(self.batch_size * self.select_experts_num, device=self.device) / self.select_experts_num
        else:
            self.origin_gate_weight = gate_weight.flatten()


        self._count_experts()
        self.a2a_time = 0

        """
        experts_traffic是对当前通信下所有experts的发送token数量的统计，没有token发送的experts就记为0
        gpu_traffic是考虑往每个GPU上面发送token数量的统计，没有token发送的gpu就记为0
        如果一个gpu上面存在多个experts，那么experts_traffic和gpu_traffic就会不同，否则是相同的
        
        对于experts_layout = [2, 3, 4, 2]的情况
        此时如果experts_traffic = [10, 20, 12, 41, 22, 18, 24, 27, 9, 16, 11]
        那么gpu_traffic_split = [[10, 20], [12, 41, 22], [18, 24, 27, 9], [16, 11]]
        gpu_traffic = [30, 75, 78, 27]代表每个卡上面分配的token数量
        首先通过all2all通信发送的gpu_split,让每个卡知道自己的每个expert会被分配到多少token
        这样才能在交换数据的时候提前开辟同样大小的空间来存储,否则会报错
        """
        # 统计每个expert经过的token数量
        self.experts_traffic.zero_()
        self.experts_traffic[self.select_experts_id] = self.experts_id_counts

        # 统计每个GPU分配的token数量(把每个gpu下expert的token数量相加)
        gpu_traffic_split = torch.split(self.experts_traffic, self.experts_layout)
        for idx, gpu in enumerate(gpu_traffic_split):
            self.gpu_traffic[idx] = gpu.sum()

        # 先使用all2all将每个gpu上面的每个expert会分到多少样本发送出去
        a2a_start_time = time.perf_counter()
        handle = dist.all_to_all(output_tensor_list=self.num4experts_from_gpu_list, input_tensor_list=list(gpu_traffic_split), async_op=True)
        handle.wait()
        self.a2a_time += (time.perf_counter() - a2a_start_time)

        """
        num4experts_from_gpu_list是一个列表，里面每个元素是一个tensor，代表当前GPU上面的每个expert从其他GPU被分配到的token数量。
        num4gpu_from_gpu_list也是一个列表，里面每个元素是一个大小为1的tensor，代表当前GPU从其他GPU被分配到的总token数量。
        
        如果每个GPU上面只有一个expert,那么这两个list是相同的。
        """
        # self.num4experts_from_gpu_list = self.gpu_traffic_output_list
        self.num4gpu_from_gpu_list = [x.sum().item() for x in self.num4experts_from_gpu_list]
        temp = torch.cat(self.num4experts_from_gpu_list).reshape(self.gpu_num, self.local_experts_num)
        temp = temp.T
        temp_flatten = temp.flatten()
        self.e2g_convert_list = list(temp_flatten.chunk(self.local_experts_num))

        # 站在experts角度，对数据进行划分，每份数据来自哪个GPU
        # self.combine_split = torch.cat(self.num4experts_from_gpu_list)
        self.prepare_time = 0


    def _count_experts(self):
        experts_id_flatten = self.expert_id.detach().clone().flatten()
        batch_flatten = torch.arange(self.batch_size * self.select_experts_num, device=self.device).div_(self.select_experts_num, rounding_mode='trunc')

        # 通过sort操作将相同expert的聚集在一起，并根据这个索引获取batch中每个样本的位置变化
        sort_values, self.experts_change_indices = experts_id_flatten.sort(0)
        self.batch_change_indices = batch_flatten[self.experts_change_indices]
        self.change_gate_weight = self.origin_gate_weight[self.batch_change_indices]

        # 获取本次路由的结果，有哪些experts被选中了，并统计了每个expert被选中的次数
        select_experts_id, experts_id_counts = torch.unique_consecutive(sort_values, return_counts=True)
        self.select_experts_id = select_experts_id
        self.experts_id_counts = experts_id_counts


    # 将每个卡上面的tensor_list进行all2all通信
    def _simple_all2all(self, tensor_list, output_split_list, outsize):
        total_sample = sum(output_split_list)
        output_list = list(torch.split(torch.zeros((total_sample, outsize), dtype=torch.float32, device=self.device, requires_grad=True), output_split_list))
        t1 = time.perf_counter()
        handle = dist.all_to_all(output_tensor_list=output_list, input_tensor_list=tensor_list, async_op=True)
        handle.wait()
        self.a2a_time += (time.perf_counter() - t1)
        return output_list

    def _rearrange_data(self, experts_input, mode='g2e'):
        """
        该函数适用于一个GPU上面有多个expert的时候，如果一个GPU上面只有一个专家则不需要调用。

        对于mode == 'g2e'，配合第一个all2all的通信过程。GPU并行转专家并行时，GPU通过all2all通信之后，数据是按照GPU来进行划分的
        每份GPU的数据里面都有属于同一个expert的数据，此时需要将这些数据聚合在一起(可以理解为将GPU维度划分的数据变为expert维度划分)

        对于mode == 'e2g'，配合第二个all2all的通信过程。专家并行转GPU并行时，GPU通过all2all通信之前，数据是按照expert来进行划分的
        每份expert的数据里面都有属于同一个GPU的数据，此时需要将这些数据聚合在一起(可以理解为将expert维度划分的数据变为GPU维度划分)
        """
        assert mode == 'g2e' or mode == 'e2g', "parameter 'mode' should be 'g2e' or 'e2g'"
        if mode == 'g2e':
            loop1_num = self.local_experts_num
            loop2_num = self.gpu_num
            loop_list = self.num4experts_from_gpu_list
            experts_input = torch.split(experts_input, self.num4gpu_from_gpu_list)
        elif mode == 'e2g':
            loop1_num = self.gpu_num
            loop2_num = self.local_experts_num
            loop_list = self.e2g_convert_list

        adjust_output_list = []
        split_list = []
        for loop1_id in range(loop1_num):
            temp = []
            for loop2_id in range(loop2_num):
                if mode == 'g2e':
                    k = sum(self.num4experts_from_gpu_list[loop2_id][:loop1_id])
                    temp.append(experts_input[loop2_id][k:k + self.num4experts_from_gpu_list[loop2_id][loop1_id]])
                else:
                    split_by_gpu = list(torch.split(experts_input[loop2_id], loop_list[loop2_id].tolist()))
                    temp.append(split_by_gpu[loop1_id])
            temp_input = torch.cat(temp, 0)
            adjust_output_list.append(temp_input)
            split_list.append(len(temp_input))
        return torch.cat(adjust_output_list, 0), torch.tensor(split_list, device=self.device)

    def dispatch(self, experts_in, compressor=None):
        dispatch_start_time = time.perf_counter()

        self.input_size = experts_in.shape[-1]
        assert self.batch_size == experts_in.shape[0], \
        "error, input data has {} in batch size, expected batch size {}".format(experts_in.shape[0], self.batch_size)

        # 对数据进行压缩
        if compressor != None:
            experts_in = compressor.compress(experts_in, [])

        # 根据batch中每个样本位置的变化，重排变换后的batch，并按照发往每个gpu的数量进行split
        experts_data = experts_in[self.batch_change_indices].squeeze(1)
        tensor_list = list(torch.split(experts_data, self.gpu_traffic.tolist(), dim=0))
        experts_input = self._simple_all2all(tensor_list, self.num4gpu_from_gpu_list, self.input_size)

        experts_input = torch.cat(experts_input, 0)
        experts_input = Stitcher.apply(experts_data, experts_input, torch.tensor(self.num4gpu_from_gpu_list), self.gpu_traffic)
        if self.local_experts_num > 1:
            experts_input, experts_split = self._rearrange_data(experts_input, mode='g2e')
            experts_input = torch.split(experts_input, experts_split.tolist())
        else:
            experts_input = [experts_input]
        self.dispatch_time = time.perf_counter() - dispatch_start_time

        return experts_input

    def combine(self, experts_out):
        combine_start_time = time.perf_counter()
        # experts_out是一个列表，里面的每一个元素都是一个expert输出的结果
        self.output_size = experts_out[0].shape[1]

        if self.local_experts_num > 1:
            experts_out, experts_split = self._rearrange_data(experts_out, mode='e2g')
        else:
            experts_out, experts_split = experts_out[0], torch.cat(self.num4experts_from_gpu_list)

        tensor_list = list(torch.split(experts_out, experts_split.tolist(), dim=0))
        experts_traffic_input_list = list(torch.split(self.gpu_traffic, 1))
        compute_res = self._simple_all2all(tensor_list, experts_traffic_input_list, self.output_size)
        compute_res = torch.cat(compute_res, 0)
        compute_res = Stitcher.apply(experts_out, compute_res, self.gpu_traffic, experts_split)
        compute_res = compute_res * self.change_gate_weight.reshape(-1, 1)

        zeros = torch.zeros(self.batch_size, self.output_size, requires_grad=True, device=self.device)
        combined = zeros.index_add(0, self.batch_change_indices, compute_res)
        self.combine_time = time.perf_counter() - combine_start_time
        return combined


# MOE层
class Moe_layer(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size, k=1, host_num=1, device='cpu'):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size
        self.experts_layout = num_experts
        self._hidden_size = hidden_size
        self.k = k
        self.rank = dist.get_rank()
        self.device = device

        # 确定experts在每个GPU上面的布局
        if type(num_experts) == list:
            self.local_experts_num = num_experts[self.rank]
            self.global_experts_num = sum(num_experts)
            self.gpu_num = len(num_experts)
        else:
            self.local_experts_num = 1
            self.global_experts_num = num_experts
            self.gpu_num = num_experts

        # self.gpu_num_per_host = int(self.gpu_num / self.host_num)
        # self.current_host_id = int(self.rank / self.gpu_num_per_host)

        """
        如果有两个host，每个host有两个GPU，如果在每个GPU上面存放25个专家
        那么experts_id_layout = [25, 50, 75, 100]
        """

        self.experts_id_layout = []

        experts_id_count = 0
        for experts_number in self.experts_layout:
            experts_id_count += experts_number
            self.experts_id_layout.append(experts_id_count)

        # self.local_streams_list = [cuda.Stream() for _ in range(self.local_experts_num)]

        # 创建gate
        self.gater = TopKGate(input_size, self.global_experts_num, k, device)
        # self.host_gate = TopKGate(input_size, self.host_num, 1, device=device)

        # 创建experts
        self.base_expert_fc1 = torch.zeros(hidden_size, input_size, device=self.device, requires_grad=False)
        self.base_expert_fc2 = torch.zeros(output_size, hidden_size, device=self.device, requires_grad=False)
        nn.init.kaiming_uniform(self.base_expert_fc1)
        nn.init.kaiming_uniform(self.base_expert_fc2)

        # del self.base_expert
        self.experts = nn.ModuleList([Expert(input_size, output_size, hidden_size) for _ in range(self.local_experts_num)]).to(device)

        # 初始化每个GPU和每个expert的流量统计
        self.experts_throughput = torch.zeros(self.global_experts_num)
        self.gpu_throughput = torch.zeros(self.gpu_num)

        # self.compressor = TopKCompressor(0.1, device=device)
        self.compressor = NoneCompressor()

        self.a2a_manager = All2AllManager(experts_layout=self.experts_layout, device=self.device)

        self.non_moe_state = 0


    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])

        if self.non_moe_state > 0:
            self.non_moe_state -= 1
            input_data = self.compressor.decompress(x)
            result = self.base_expert(input_data)
        else:
            experts_gate, experts_id = self.gater(x)
            # all2all通信，experts_input即为按照experts进行划分的数据，experts_select_id即为现在这个batch计算出来的需要发送的experts编号(可能不是所有experts被激活)，和前面的experts_input一一对应
            self.a2a_manager.connect(expert_id=experts_id, gate_weight=experts_gate)

            self.experts_throughput += self.a2a_manager.experts_traffic.cpu()
            self.gpu_throughput += self.a2a_manager.gpu_traffic.cpu()
            # print("experts throughput: {}".format(self.experts_throughput.tolist()))
            # print("gpu throughput: {}".format(self.gpu_throughput))
            """####### 前向传播dispatch的all to all过程的压缩###########"""
            x = disp_comp.apply(x, level, experts_id)
            """######################################################"""
            expert_in = self.a2a_manager.dispatch(x, compressor=self.compressor)  # 输入：该节点下的token发到不同专家下
            expert_out = []                                                       # 返回：该节点下的k个专家收到了哪些token
            for expert_id in range(self.local_experts_num):
                output=self.experts[expert_id](expert_in[expert_id])
                """######### 前向传播combine的all to all过程的压缩 ########"""
                if expert_in[expert_id].numel() != 0:   output = comb_comp.apply(output, level)
                """#####################################################"""
                expert_out.append(output)
            result = self.a2a_manager.combine(expert_out)
        return result.reshape(shape)









# import torch
# import torch.nn as nn
# from torch.autograd.function import Function
# import torch.distributed as dist
# import time
# import torch.cuda as cuda
# import math
# # import torch.quantization as quant


# def random_mask(size, mask_ratio=0, device='cpu'):
#     total_size = torch.prod(torch.tensor(size)).item()
#     threshold = math.ceil(total_size * mask_ratio)
#     temp = torch.randperm(total_size)
#     mask_idx = (temp < threshold)
#     mask_result = torch.ones(total_size, device=device)
#     mask_result[mask_idx] = 0
#     return mask_result.reshape(size)


# # 定义专家模型
# class Expert(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.relu = nn.ReLU()
#         # self.log_soft = nn.LogSoftmax(1)

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         # out = self.log_soft(out)
#         return out

# # 定义gate，使用topk
# class TopKGate(nn.Module):
#     def __init__(self, input_size, num_experts, k, device='cpu'):
#         super().__init__()
#         self._input_size = input_size
#         self.num_experts = num_experts
#         self.k = k
#         assert k >= 1 and k <= num_experts, \
#         "k should satisfy: k >= 1 and k <= total experts number"
#         self.device = device

#         self.gate_layer = nn.Linear(input_size, num_experts, bias=True, device=device)
#         self.smooth_func = nn.Softmax(dim=1)
#         self.count = 0

#     def gating(self, x):
#         gate_res = self.gate_layer(x)
#         topk_gate, topk_index = gate_res.topk(self.k, dim=1)
#         return self.smooth_func(topk_gate), topk_index

#     def forward(self, x):
#         return self.gating(x)


# # 定义topk压缩器
# class TopKCompressor():
#     def __init__(self, sparse_ratio, device='cpu') -> None:
#         self.sparse_ratio = sparse_ratio
#         self.device = device

#     def compress(self, tensor, ctx=[]):
#         shape = tensor.shape
#         dtype = tensor.dtype
#         ctx = shape, dtype
#         return tensor, ctx

#     def decompress(self, tensor, ctx=[]):
#         shape, dtype = ctx
#         decompressed_tensor = torch.zeros(shape, device=self.device,dtype=dtype)
#         values, indices = tensor
#         decompressed_tensor[indices].data = values.data
#         return decompressed_tensor

# # 无压缩器
# class NoneCompressor():
#     def __init__(self):
#         pass

#     def compress(self, tensor, ctx=[]):
#         return tensor

#     def decompress(self, tensor, ctx=[]):
#         return tensor



# class Stitcher(Function):
#     @staticmethod
#     def forward(ctx, a2a_input_tensor, a2a_output_tensor, b2e_split, e2b_split):
#         """
#         对于第一个all2all过程，a2a_input_tensor代表batch划分的数据，input_split是发往每个expert的样本个数。
#             a2a_output_tensor代表通信交换完成的tensor，对应当前进程expert需要的输入。

#         对于第二个all2all过程，a2a_input_tensor代表expert划分的数据，output_split是发往每个worker的样本个数。
#             a2a_output_tensor代表通信交换完成的tensor，对应的是当前worker需要的输入(后面求loss等)。

#         该类是建立all2all通信前后tensor的联系，使得反向传播的时候计算图可以连接起来。
#         """
#         ctx.save_for_backward(b2e_split, e2b_split)
#         return a2a_output_tensor

#     @staticmethod
#     def backward(ctx, grad_output):
#         b2e_split, e2b_split = ctx.saved_tensors
#         device = grad_output.device

#         b2e_split = b2e_split.tolist()
#         e2b_split = e2b_split.tolist()

#         input_grad_list = list(torch.split(grad_output, b2e_split, dim=0))
#         total_sample = sum(e2b_split)
#         output_grad_list = list(torch.split(torch.zeros((total_sample, grad_output.shape[1]), device=device), e2b_split, dim=0))
#         handle = dist.all_to_all(output_tensor_list=output_grad_list, input_tensor_list=input_grad_list, async_op=True)
#         handle.wait()

#         temp_grad = torch.cat(output_grad_list)
#         return temp_grad, None, None, None


# # all2all转换器，连接在gate和experts之间
# class All2AllManager():
#     """
#     Args:
#         expert_id: the selected experts id, with size (batch_size, selected_experts_number)
#         experts_num: the total number of experts or the list of the global experts layout
#         gate_weight: the gate weight for each experts output
#         device: the running device
#     """
#     def __init__(self, experts_layout, device='cpu'):
#         self.rank = dist.get_rank()
#         self.device = device

#         """
#         如果experts_layout是按照列表形式给出,如experts_layout = [2, 3, 4, 2],则代表四个卡参加,每个卡上面2,3,4,2个experts
#         如果是一个数字的话,则有这么多卡参加训练,默认一个卡上面一个expert
#         """
#         if type(experts_layout) == list:
#             self.global_experts_num = sum(experts_layout)
#             self.gpu_num = len(experts_layout)
#             self.local_experts_num = experts_layout[self.rank]
#             self.experts_layout = experts_layout
#         else:
#             self.global_experts_num = experts_layout
#             self.gpu_num = experts_layout
#             self.local_experts_num = 1
#             self.experts_layout = [1] * experts_layout

#         self.experts_traffic = torch.zeros(self.global_experts_num, device=self.device, dtype=torch.long)
#         self.gpu_traffic = torch.zeros(self.gpu_num, device=self.device, dtype=torch.long)

#         # 在第一次dispatch之前需要先知道每个expert会处理多少token
#         self.num4experts_from_gpu_list = list(torch.zeros(self.local_experts_num * self.gpu_num, device=self.device, dtype=torch.long).chunk(self.gpu_num))



#     def connect(self, expert_id, gate_weight=None):
#         self.expert_id = expert_id
#         self.batch_size, self.select_experts_num = expert_id.shape

#         if gate_weight == None:
#             self.origin_gate_weight = torch.ones(self.batch_size * self.select_experts_num, device=self.device) / self.select_experts_num
#         else:
#             self.origin_gate_weight = gate_weight.flatten()

#         # # 获取本次路由的结果，有哪些experts被选中了，并统计了每个expert被选中的次数

#         # # experts_traffic是对当前通信下所有experts的发送token数量的统计，没有token发送的experts就记为0

#         # # 站在每个GPU角度，对数据进行划分，每份数据发给哪个expert

#         self._count_experts()
#         self.a2a_time = 0

#         """
#         experts_traffic是对当前通信下所有experts的发送token数量的统计，没有token发送的experts就记为0
#         gpu_traffic是考虑往每个GPU上面发送token数量的统计，没有token发送的gpu就记为0
#         如果一个gpu上面存在多个experts，那么experts_traffic和gpu_traffic就会不同，否则是相同的

#         对于experts_layout = [2, 3, 4, 2]的情况
#         此时如果experts_traffic = [10, 20, 12, 41, 22, 18, 24, 27, 9, 16, 11]
#         那么gpu_traffic_split = [[10, 20], [12, 41, 22], [18, 24, 27, 9], [16, 11]]
#         gpu_traffic = [30, 75, 78, 27]代表每个卡上面分配的token数量
#         首先通过all2all通信发送的gpu_split,让每个卡知道自己的每个expert会被分配到多少token
#         这样才能在交换数据的时候提前开辟同样大小的空间来存储,否则会报错
#         """
#         # 统计每个expert经过的token数量
#         self.experts_traffic.zero_()
#         self.experts_traffic[self.select_experts_id] = self.experts_id_counts

#         # 统计每个GPU分配的token数量
#         gpu_traffic_split = torch.split(self.experts_traffic, self.experts_layout)
#         for idx, gpu in enumerate(gpu_traffic_split):
#             self.gpu_traffic[idx] = gpu.sum()

#         # 先使用all2all将每个gpu上面的每个expert会分到多少样本发送出去
#         a2a_start_time = time.perf_counter()
#         handle = dist.all_to_all(output_tensor_list=self.num4experts_from_gpu_list, input_tensor_list=list(gpu_traffic_split), async_op=True)
#         handle.wait()
#         self.a2a_time += (time.perf_counter() - a2a_start_time)

#         """
#         num4experts_from_gpu_list是一个列表，里面每个元素是一个tensor，代表当前GPU上面的每个expert从其他GPU被分配到的token数量。
#         num4gpu_from_gpu_list也是一个列表，里面每个元素是一个大小为1的tensor，代表当前GPU从其他GPU被分配到的总token数量。

#         如果每个GPU上面只有一个expert,那么这两个list是相同的。
#         """
#         # self.num4experts_from_gpu_list = self.gpu_traffic_output_list
#         self.num4gpu_from_gpu_list = [x.sum().item() for x in self.num4experts_from_gpu_list]
#         temp = torch.cat(self.num4experts_from_gpu_list).reshape(self.gpu_num, self.local_experts_num)
#         temp = temp.T
#         temp_flatten = temp.flatten()
#         self.e2g_convert_list = list(temp_flatten.chunk(self.local_experts_num))

#         # 站在experts角度，对数据进行划分，每份数据来自哪个GPU
#         # self.combine_split = torch.cat(self.num4experts_from_gpu_list)
#         self.prepare_time = 0


#     def _count_experts(self):
#         experts_id_flatten = self.expert_id.detach().clone().flatten()
#         batch_flatten = torch.arange(self.batch_size * self.select_experts_num, device=self.device).div_(self.select_experts_num, rounding_mode='trunc')

#         # 通过sort操作将相同expert的聚集在一起，并根据这个索引获取batch中每个样本的位置变化
#         sort_values, self.experts_change_indices = experts_id_flatten.sort(0)
#         self.batch_change_indices = batch_flatten[self.experts_change_indices]
#         self.change_gate_weight = self.origin_gate_weight[self.batch_change_indices]

#         # 获取本次路由的结果，有哪些experts被选中了，并统计了每个expert被选中的次数
#         select_experts_id, experts_id_counts = torch.unique_consecutive(sort_values, return_counts=True)
#         self.select_experts_id = select_experts_id
#         self.experts_id_counts = experts_id_counts


#     # 将每个卡上面的tensor_list进行all2all通信
#     def _simple_all2all(self, tensor_list, output_split_list, outsize):
#         total_sample = sum(output_split_list)
#         output_list = list(torch.split(torch.zeros((total_sample, outsize), dtype=torch.float32, device=self.device, requires_grad=True), output_split_list))
#         t1 = time.perf_counter()
#         handle = dist.all_to_all(output_tensor_list=output_list, input_tensor_list=tensor_list, async_op=True)
#         handle.wait()
#         self.a2a_time += (time.perf_counter() - t1)
#         return output_list

#     def _rearrange_data(self, experts_input, mode='g2e'):
#         """
#         该函数适用于一个GPU上面有多个expert的时候，如果一个GPU上面只有一个专家则不需要调用。

#         对于mode == 'g2e'，配合第一个all2all的通信过程。GPU并行转专家并行时，GPU通过all2all通信之后，数据是按照GPU来进行划分的
#         每份GPU的数据里面都有属于同一个expert的数据，此时需要将这些数据聚合在一起(可以理解为将GPU维度划分的数据变为expert维度划分)

#         对于mode == 'e2g'，配合第二个all2all的通信过程。专家并行转GPU并行时，GPU通过all2all通信之前，数据是按照expert来进行划分的
#         每份expert的数据里面都有属于同一个GPU的数据，此时需要将这些数据聚合在一起(可以理解为将expert维度划分的数据变为GPU维度划分)
#         """
#         assert mode == 'g2e' or mode == 'e2g', "parameter 'mode' should be 'g2e' or 'e2g'"
#         if mode == 'g2e':
#             loop1_num = self.local_experts_num
#             loop2_num = self.gpu_num
#             loop_list = self.num4experts_from_gpu_list
#             experts_input = torch.split(experts_input, self.num4gpu_from_gpu_list)
#         elif mode == 'e2g':
#             loop1_num = self.gpu_num
#             loop2_num = self.local_experts_num
#             loop_list = self.e2g_convert_list

#         adjust_output_list = []
#         split_list = []
#         for loop1_id in range(loop1_num):
#             temp = []
#             for loop2_id in range(loop2_num):
#                 if mode == 'g2e':
#                     k = sum(self.num4experts_from_gpu_list[loop2_id][:loop1_id])
#                     temp.append(experts_input[loop2_id][k:k + self.num4experts_from_gpu_list[loop2_id][loop1_id]])
#                 else:
#                     split_by_gpu = list(torch.split(experts_input[loop2_id], loop_list[loop2_id].tolist()))
#                     temp.append(split_by_gpu[loop1_id])
#             temp_input = torch.cat(temp, 0)
#             adjust_output_list.append(temp_input)
#             split_list.append(len(temp_input))
#         return torch.cat(adjust_output_list, 0), torch.tensor(split_list, device=self.device)

#     def dispatch(self, experts_in, compressor=None):
#         dispatch_start_time = time.perf_counter()

#         self.input_size = experts_in.shape[-1]
#         assert self.batch_size == experts_in.shape[0], \
#         "error, input data has {} in batch size, expected batch size {}".format(experts_in.shape[0], self.batch_size)

#         # 对数据进行压缩
#         if compressor != None:
#             experts_in = compressor.compress(experts_in, [])

#         # 根据batch中每个样本位置的变化，重排变换后的batch，并按照发往每个gpu的数量进行split
#         experts_data = experts_in[self.batch_change_indices].squeeze(1)
#         tensor_list = list(torch.split(experts_data, self.gpu_traffic.tolist(), dim=0))
#         experts_input = self._simple_all2all(tensor_list, self.num4gpu_from_gpu_list, self.input_size)

#         experts_input = torch.cat(experts_input, 0)
#         experts_input = Stitcher.apply(experts_data, experts_input, torch.tensor(self.num4gpu_from_gpu_list), self.gpu_traffic)
#         if self.local_experts_num > 1:
#             experts_input, experts_split = self._rearrange_data(experts_input, mode='g2e')
#             experts_input = torch.split(experts_input, experts_split.tolist())
#         else:
#             experts_input = [experts_input]
#         self.dispatch_time = time.perf_counter() - dispatch_start_time
#         return experts_input

#     def combine(self, experts_out):
#         combine_start_time = time.perf_counter()
#         # experts_out是一个列表，里面的每一个元素都是一个expert输出的结果
#         self.output_size = experts_out[0].shape[1]

#         if self.local_experts_num > 1:
#             experts_out, experts_split = self._rearrange_data(experts_out, mode='e2g')
#         else:
#             experts_out, experts_split = experts_out[0], torch.cat(self.num4experts_from_gpu_list)

#         tensor_list = list(torch.split(experts_out, experts_split.tolist(), dim=0))
#         experts_traffic_input_list = list(torch.split(self.gpu_traffic, 1))
#         compute_res = self._simple_all2all(tensor_list, experts_traffic_input_list, self.output_size)
#         compute_res = torch.cat(compute_res, 0)
#         compute_res = Stitcher.apply(experts_out, compute_res, self.gpu_traffic, experts_split)
#         compute_res = compute_res * self.change_gate_weight.reshape(-1, 1)

#         zeros = torch.zeros(self.batch_size, self.output_size, requires_grad=True, device=self.device)
#         combined = zeros.index_add(0, self.batch_change_indices, compute_res)
#         self.combine_time = time.perf_counter() - combine_start_time
#         return combined

# # MOE层
# class Moe_layer(nn.Module):
#     def __init__(self, input_size, output_size, num_experts, hidden_size, k=1, host_num=1, device='cpu', mask_ratio=0, use_mask=True):
#         super().__init__()
#         self._input_size = input_size
#         self._output_size = output_size
#         self.experts_layout = num_experts
#         self._hidden_size = hidden_size
#         self.k = k
#         self.rank = dist.get_rank()
#         self.device = device
#         self.use_mask = use_mask
#         self.mask_ratio = mask_ratio
#         print("mask ratio ", mask_ratio)

#         # 确定experts在每个GPU上面的布局
#         if type(num_experts) == list:
#             self.local_experts_num = num_experts[self.rank]
#             self.global_experts_num = sum(num_experts)
#             self.gpu_num = len(num_experts)
#         else:
#             self.local_experts_num = 1
#             self.global_experts_num = num_experts
#             self.gpu_num = num_experts

#         # self.gpu_num_per_host = int(self.gpu_num / self.host_num)
#         # self.current_host_id = int(self.rank / self.gpu_num_per_host)

#         """
#         如果有两个host，每个host有两个GPU，如果在每个GPU上面存放25个专家
#         那么experts_id_layout = [25, 50, 75, 100]
#         """

#         self.experts_id_layout = []

#         experts_id_count = 0
#         for experts_number in self.experts_layout:
#             experts_id_count += experts_number
#             self.experts_id_layout.append(experts_id_count)

#         self.local_streams_list = [cuda.Stream() for _ in range(self.local_experts_num)]

#         # 创建gate
#         self.gater = TopKGate(input_size, self.global_experts_num, k, device)
#         # self.host_gate = TopKGate(input_size, self.host_num, 1, device=device)

#         # 创建experts
#         # self.base_expert = Expert(input_size, output_size, hidden_size).to(device)
#         # nn.init.kaiming_uniform(self.base_expert.fc1.weight.data)
#         # nn.init.kaiming_uniform(self.base_expert.fc2.weight.data)
#         # self.update_base_expert()
#         # self.base_expert_fc1 = self.base_expert.fc1.weight.detach().clone().requires_grad_(False).to(self.device)
#         # self.base_expert_fc2 = self.base_expert.fc2.weight.detach().clone().requires_grad_(False).to(self.device)
#         self.base_expert_fc1 = torch.zeros(hidden_size, input_size, device=self.device)
#         self.base_expert_fc2 = torch.zeros(output_size, hidden_size, device=self.device)
#         nn.init.kaiming_uniform(self.base_expert_fc1)
#         nn.init.kaiming_uniform(self.base_expert_fc2)

#         # del self.base_expert
#         self.experts = nn.ModuleList([Expert(input_size, output_size, hidden_size) for _ in range(self.local_experts_num)]).to(device)

#         # 初始化每个GPU和每个expert的流量统计
#         self.experts_throughput = torch.zeros(self.global_experts_num)
#         self.gpu_throughput = torch.zeros(self.gpu_num)

#         # self.compressor = TopKCompressor(0.1, device=device)
#         self.compressor = NoneCompressor()

#         self.a2a_manager = All2AllManager(experts_layout=self.experts_layout, device=self.device)

#         self.mask_life_cycle = 4000
#         self.mask_remain_life = 0
#         self.non_moe_state = 0

#         if use_mask:
#             self.local_mask = self._generate_local_mask(mask_ratio=mask_ratio, method='fisher')




#     def forward(self, x):
#         shape = x.shape
#         x = x.reshape(-1, shape[-1])

#         # if 0 == self.mask_remain_life:
#         #     self.local_mask = self._generate_local_mask()
#         #     self._synchronize_global_mask()
#         #     self.mask_remain_life = self.mask_life_cycle

#         # self.mask_remain_life -= 1

#         if self.non_moe_state > 0:
#             self.non_moe_state -= 1
#             input_data = self.compressor.decompress(x)
#             result = self.base_expert(input_data)
#         else:
#             # 对输入数据进行gate，返回每个数据需要被发送的expert id
#             experts_gate, experts_id = self.gater(x)

#             # all2all通信，experts_input即为按照experts进行划分的数据，experts_select_id即为现在这个batch计算出来的需要发送的experts编号(可能不是所有experts被激活)，和前面的experts_input一一对应
#             self.a2a_manager.connect(expert_id=experts_id, gate_weight=experts_gate)

#             self.experts_throughput += self.a2a_manager.experts_traffic.cpu()
#             self.gpu_throughput += self.a2a_manager.gpu_traffic.cpu()
#             # print("experts throughput: {}".format(self.experts_throughput.tolist()))
#             # print("gpu throughput: {}".format(self.gpu_throughput))

#             expert_in = self.a2a_manager.dispatch(x, compressor=self.compressor)

#             expert_out = []
#             idx = 0
#             for expert_id, local_stream in enumerate(self.local_streams_list):
#                 # with cuda.stream(local_stream):
#                 input_data = self.compressor.decompress(expert_in[expert_id])
#                 if self.use_mask:
#                     # print(self.base_expert_fc1[0][0])
#                     # print(self.base_expert_fc2[0][0])
#                     self.experts[expert_id].fc1.weight.data = self.base_expert_fc1 * self.local_mask[idx]
#                     self.experts[expert_id].fc2.weight.data = self.base_expert_fc2 * self.local_mask[idx + 1]
#                     idx += 2
#                 expert_out.append(self.experts[expert_id](input_data))


#             result = self.a2a_manager.combine(expert_out)
#         return result.reshape(shape)






#     def _generate_local_mask(self, mask_ratio=0, method='random'):
#         print("generate new local mask")
#         local_mask_list = []
#         if method == 'random':
#             for _ in range(self.local_experts_num):
#                 local_mask_list.append(random_mask((self._hidden_size, self._input_size), mask_ratio=mask_ratio, device=self.device))
#                 local_mask_list.append(random_mask((self._input_size, self._hidden_size), mask_ratio=mask_ratio, device=self.device))
#         elif method == "fisher":
#             for _ in range(self.local_experts_num):
#                 local_mask_list.append(random_mask((self._hidden_size, self._input_size), mask_ratio=0, device=self.device))
#                 local_mask_list.append(random_mask((self._input_size, self._hidden_size), mask_ratio=0, device=self.device))
#         return local_mask_list




#     def _synchronize_global_mask(self):
#         print("synchronize mask with other host")
#         pass
