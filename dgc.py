import torch
import torch.distributed as dist
import math
import random


# code modified from https://github.com/sands-lab/grace/blob/master/grace_dl/torch/memory/dgc.py
class Memory:
    @staticmethod
    def initialize(*args, **kwargs):
        pass

    @staticmethod
    def compensate(tensor, *args, **kwargs):
        return tensor
    
    @staticmethod
    def update(*args, **kwargs):
        pass

    @staticmethod
    def state_dict():
        return None
    
    @staticmethod
    def load_state_dict(state_dict):
        pass


class DGCSGDMemory(Memory):
    """ Memory for momentum correction in DGC for momentum SGD optimizer"""
    def __init__(self, momentum=0.9, nesterov=False,
                 gradient_clipping=None, momentum_masking=True):
        self.gradient_clipping = gradient_clipping
        self.momentum_masking = momentum_masking

        self.momentum = momentum
        self.nesterov = nesterov
        self.momentums = {}
        self.velocities = {}
        try:
            self.rank = dist.get_rank()
        except:
            self.rank = 0
    
    def initialize(self, named_parameters):
        if self.rank == 0:pass
            #print("=> initializing dgc sgd memory")
        for name, param in named_parameters:
            self.momentums[name] = torch.zeros_like(param.data)
            self.velocities[name] = torch.zeros_like(param.data)

    def compensate(self, grad, name, accumulate=True):
        """Update the velocities with the momentums."""
        if self.gradient_clipping is not None:
            grad = self.gradient_clipping(grad)
        mmt = self.momentums[name]
        if accumulate:
            vec = self.velocities[name]
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                vec.add_(mmt).add_(grad)
            else:
                mmt.mul_(self.momentum).add_(grad)
                vec.add_(mmt)
            return vec
        else:
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                return mmt.add(grad)
            else:
                mmt.mul_(self.momentum).add_(grad)
                return mmt.clone()  # TODO: save this clone

    def update(self, name, ctx):
        """Update the momentums."""
        indices = ctx[0]
        if self.momentum_masking:
            self.momentums[name].view(-1).index_fill_(0, indices, 0)
        self.velocities[name].view(-1).index_fill_(0, indices, 0)

    def state_dict(self):
        return dict(momentums=self.momentums, velocities=self.velocities)

    def load_state_dict(self, state_dict):
        momentums = state_dict['momentums']
        velocities = state_dict['velocities']
        for name in self.momentums.keys():
            if name in momentums:
                self.momentums[name] = momentums[name]
                self.velocities[name] = velocities[name]





class DGCCompressor:
    def __init__(self, compress_ratio, memory=None,
                 sample_ratio=2, strided_sample=True,
                 compress_upper_bound=1.3, compress_lower_bound=0.8, max_adaptation_iters=10, resample=True,
                 fp16_values=False, int32_indices=False,
                 warmup_epochs=-1, warmup_coeff=None):
        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        except:
            self.world_size = 1
            self.rank = 0
        self.op = "Average"
        self.fp16_values = fp16_values
        self.int32_indices = int32_indices

        self.base_compress_ratio = self.compress_ratio = \
            compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio
        self.memory = Memory if memory is None else memory
        self.warmup_epochs = warmup_epochs
        if self.warmup_epochs > 0:
            if warmup_coeff is None:
                self.warmup_coeff = self.base_compress_ratio \
                                    ** (1. / (self.warmup_epochs + 1))
            else:
                if isinstance(warmup_coeff, (tuple, list)):
                    assert len(warmup_coeff) >= self.warmup_epochs
                    for wc in warmup_coeff:
                        assert 0 < wc <= 1
                else:
                    assert 0 < warmup_coeff <= 1
                self.warmup_coeff = warmup_coeff
        else:
            self.warmup_coeff = 1

        self.sample_ratio = min(max(sample_ratio, 0.01), 1.0)
        self.strided_sample = strided_sample
        self.compress_upper_bound = compress_upper_bound
        self.compress_lower_bound = compress_lower_bound
        self.max_adaptation_iters = max_adaptation_iters
        self.resample = resample

        self.attributes = {}

    def initialize(self, named_parameters):
        if self.rank == 0:pass
            #print("=> initializing dgc compressor")
        for name, param in named_parameters:
            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]
            if self.sample_ratio < 1.0:
                pct_numel = int(math.ceil(numel * self.sample_ratio))
                cpr_numel = int(math.ceil(2 / self.compress_ratio))
                if numel <= cpr_numel:
                    if self.rank == 0:pass
                        #print(f'Warning: {name} with {numel} elements transmits 1 gradient element')
                    sample_stride = 1
                    num_samples = numel
                else:
                    sample_stride = int(math.ceil(numel / max(pct_numel, cpr_numel) / 32)) * 32 + 1
                    num_samples = numel // sample_stride
                    while num_samples < max(pct_numel, cpr_numel):
                        sample_stride = sample_stride - 8
                        num_samples = numel // sample_stride
            else:
                sample_stride = 1
                num_samples = numel
            top_k_samples = int(math.ceil(num_samples * self.compress_ratio))
            num_selects = int(math.ceil(numel * self.compress_ratio))
            self.attributes[name] = (numel, shape, num_selects, num_samples, top_k_samples, sample_stride)
            if self.rank == 0:pass
                #print(f'   {name:<25}: transmit {num_selects} / {numel} elements of shape {shape}\n'
                #      f'   {" " * 25}  threshold {top_k_samples} / {num_samples} samples'
                #      f' {f"at stride {sample_stride}" if self.strided_sample else "uniformly"}')
    
    def warmup_compress_ratio(self, epoch):
        if self.warmup_epochs > 0:
            if epoch < self.warmup_epochs:
                if isinstance(self.warmup_coeff, (tuple, list)):
                    compress_ratio = self.warmup_coeff[epoch]
                else:
                    compress_ratio = max(self.warmup_coeff ** (epoch + 1),
                                        self.base_compress_ratio)
            else:
                compress_ratio = self.base_compress_ratio
        else:
            compress_ratio = self.base_compress_ratio
        if compress_ratio != self.compress_ratio:
            if self.rank == 0:pass
                #print(f'update compress ratio: {compress_ratio}')
            self.compress_ratio = compress_ratio
            self.initialize(self.attributes.items())

    def _sparsify(self, tensor, name):
        tensor = tensor.view(-1)
        numel, shape, num_selects, num_samples, top_k_samples, sample_stride = self.attributes[name]

        importance = tensor.abs()
        if numel == num_samples:
            samples = importance
        else:
            if self.strided_sample:
                sample_start = random.randint(0, sample_stride - 1)
                samples = importance[sample_start::sample_stride]
            else:
                samples = importance[torch.randint(0, numel, (num_samples, ), device=tensor.device)]

        threshold = torch.min(torch.topk(samples, top_k_samples, 0, largest=True, sorted=False)[0])
        mask = torch.ge(importance, threshold)
        indices = mask.nonzero().view(-1)
        num_indices = indices.numel()

        if numel > num_samples:
            # code modified from https://github.com/sands-lab/grace/blob/master/grace_dl/torch/compressor/dgc.py
            for _ in range(self.max_adaptation_iters):
                if num_indices > num_selects:
                    if num_indices > num_selects * self.compress_upper_bound:
                        if self.resample:
                            indices = indices[
                                torch.topk(importance[indices], num_selects,
                                           0, largest=True, sorted=False)[1]
                            ]
                            break
                        else:
                            threshold = threshold * self.compress_upper_bound
                    else:
                        break
                elif num_indices < self.compress_lower_bound * num_selects:
                    threshold = threshold * self.compress_lower_bound
                else:
                    break
                mask = torch.ge(importance, threshold)
                indices = mask.nonzero().view(-1)
                num_indices = indices.numel()

        indices = indices[:num_selects]
        values = tensor[indices]
        return values, indices, numel, shape, num_selects

    def compress(self, tensor, name):
        if self.compress_ratio < 1.0 and name in self.attributes:
            # compress
            tensor_compensated = self.memory.compensate(
                tensor, name, accumulate=True)
            values, indices, numel, shape, num_selects = \
                self._sparsify(tensor_compensated, name)
            self.memory.update(name, (indices, ))
            indices = indices.view(-1, 1)
            values = values.view(-1, 1)

            ctx = (name, numel, shape, values.dtype, indices.dtype,
                   tensor.data.view(numel))
            if self.fp16_values and values.dtype.is_floating_point:
                values = values.type(torch.float16)
            if self.int32_indices and not indices.dtype.is_floating_point:
                indices = indices.type(torch.int32)
            return (values, indices), ctx
        else:
            ctx = (name, None, None, tensor.dtype, None, None)
            if self.fp16_values and tensor.dtype.is_floating_point:
                tensor = tensor.type(torch.float16)
            return tensor, ctx

    def decompress(self, tensor, ctx):
        name, numel, shape, vdtype, idtype, grad = ctx
        if self.compress_ratio < 1.0 and name in self.attributes:
            assert isinstance(tensor, (list, tuple))
            values, indices = tensor
            values = values.view(-1)
            indices = indices.view(-1)
            if self.fp16_values and vdtype.is_floating_point:
                values = values.type(vdtype)
            if self.int32_indices and not idtype.is_floating_point:
                indices = indices.type(idtype)
            grad.zero_().index_put_([indices], values, accumulate=True)
            if self.op == "Average":
                grad.mul_(1. / self.world_size)
            return grad.view(shape)
        else:
            if self.fp16_values and vdtype.is_floating_point:
                tensor = tensor.type(vdtype)
            return self.memory.compensate(tensor, name, accumulate=False)

    # def communicate(self, tensor_compressed, name, op):
    #     self.op = op
    #     if self.compress_ratio < 1.0 and name in self.attributes:
    #         return [allgather_async_(t, name=f'{name}.t{e}')
    #                 for e, t in enumerate(tensor_compressed)]
    #     else:
    #         return allreduce_async_(tensor_compressed, name=name, op=op)

    # def synchronize(self, handle):
    #     if isinstance(handle, (tuple, list)):
    #         return [synchronize_(h) for h in handle]
    #     else:
    #         return synchronize_(handle)






if __name__ == "__main__":
    #print("begin test DGC")
    compressor = DGCCompressor(0.1)
    named_parameters = [("a", torch.randn(100, 100)), ("b", torch.randn(100, 200)), ("c", torch.randn(200, 100))]
    compressor.initialize(named_parameters)
    for i in range(10):
        compressor.warmup_compress_ratio(i)

    for name, param in named_parameters:
        compressed_tensor, ctx = compressor.compress(param, name)
        restored_tensor = compressor.decompress(compressed_tensor, ctx)
        print(restored_tensor.shape, param.shape)
        assert restored_tensor.shape == param.shape
    #print("end test DGC")
