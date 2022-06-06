from collections import Iterable, defaultdict
from copy import deepcopy
from itertools import chain
import numpy as np

import paddle
import torch
from paddle import nn
from torch._utils import _unflatten_dense_tensors
from torch.autograd import Variable
#from torch.nn.utils import parameters_to_vector

bn_types = (nn.BatchNorm1D, nn.BatchNorm2D, nn.BatchNorm3D)

param_count = 0
debug = 0

def split_bn_bias(layer_groups):
    "Split the layers in `layer_groups` into batchnorm (`bn_types`) and non-batchnorm groups."
    split_groups = []
    for l in layer_groups:
        l1, l2 = [], []
        for c in l.children():
            if isinstance(c, bn_types): l2.append(c)
            else: l1.append(c)
        split_groups += [nn.Sequential(*l1), nn.Sequential(*l2)]
    return split_groups

def parameters_to_vector(parameters):
    vec = []
    for param in parameters:
        vec.append(param)
    return paddle.concat(vec)

def get_master(layer_groups, flat_master: bool = False):
    "Return two lists, one for the model parameters in FP16 and one for the master parameters in FP32."
    split_groups = split_bn_bias(layer_groups)
    model_params = [[
        param for param in lg.parameters() if param.stop_gradient==False
    ] for lg in split_groups]
    if flat_master:
        master_params = []
        for lg in model_params:
            if len(lg) != 0:
                tmp = parameters_to_vector([paddle.cast(param, dtype='float32') for param in lg])
                #mp = torch.nn.Parameter(mp, requires_grad=True)
                mp = paddle.create_parameter(mp.shape, mp.dtype)
                mp.set_value(tmp)
                mp.stop_gradient=False
                if mp.grad is None: mp.grad = mp.new(*mp.size())
                master_params.append([mp])
            else:
                master_params.append([])
        return model_params, master_params
    else:
        master_params = [[paddle.cast(param.clone(), dtype='float32') for param in lg]
                         for lg in model_params]
        for mp in master_params:
            for param in mp:
                param.stop_gradient = False
        return model_params, master_params


def model_g2master_g(model_params, master_params,
                     flat_master: bool = False) -> None:
    "Copy the `model_params` gradients to `master_params` for the optimizer step."
    if flat_master:
        for model_group, master_group in zip(model_params, master_params):
            if len(master_group) != 0:
                master_group[0].grad.data.copy_(
                    parameters_to_vector(
                        [p.grad.data.float() for p in model_group]))
    else:
        for model_group, master_group in zip(model_params, master_params):
            for model, master in zip(model_group, master_group):
                if model.grad is not None:
                    if master.grad is None:
                        master.grad = master.data.new(*master.data.size())
                    master.grad.data.copy_(model.grad.data)
                else:
                    master.grad = None


def master2model(model_params, master_params,
                 flat_master: bool = False) -> None:
    "Copy `master_params` to `model_params`."
    if flat_master:
        for model_group, master_group in zip(model_params, master_params):
            if len(model_group) != 0:
                for model, master in zip(
                        model_group,
                        _unflatten_dense_tensors(master_group[0].data,
                                                 model_group)):
                    model.data.copy_(master)
    else:
        for model_group, master_group in zip(model_params, master_params):
            for model, master in zip(model_group, master_group):
                model.data.copy_(master.data)


def listify(p=None, q=None):
    "Make `p` listy and the same length as `q`."
    if p is None: p = []
    elif isinstance(p, str): p = [p]
    elif not isinstance(p, Iterable): p = [p]
    n = q if type(q) == int else len(p) if q is None else len(q)
    if len(p) == 1: p = p * n
    assert len(p) == n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


def trainable_params(m: nn.Layer):
    "Return list of trainable params in `m`."
    #res = filter(stop_gradient, m.parameters())
    res = []
    for p in m.parameters():
        if not p.stop_gradient:
            res.append(p)
    print("the len of params need grad: ", len(res))
    return res


def is_tuple(x) -> bool:
    return isinstance(x, tuple)


# copy from fastai.
class OptimWrapper(paddle.optimizer.Optimizer):
    "Basic wrapper around `opt` to simplify hyper-parameters changes."

    def __init__(self, opt, wd, true_wd: bool = False, bn_wd: bool = True):
        # super().__init__(opt.param_groups, dict())
        self.opt, self.true_wd, self.bn_wd = opt, true_wd, bn_wd
        self.opt_keys = list(self.opt._param_groups[0].keys())
        self.opt_keys.remove('params')
        self.read_defaults()
        self.wd = wd
        print("opt=", self.opt)
        print("opt_keys=", self.opt_keys)
        print("wd=", self.wd, "true_wd=", self.true_wd, "bn_wd=", self.bn_wd)

    @classmethod
    def create(cls, opt_func, lr, layer_groups, **kwargs):
        "Create an `optim.Optimizer` from `opt_func` with `lr`. Set lr on `layer_groups`."
        split_groups = split_bn_bias(layer_groups)
        #clip = paddle.nn.ClipGradByNorm(10.0)
        opt = opt_func(parameters=[{
            'params': trainable_params(l),
            'learning_rate':0,
            'beta1':0.9,
            'beta2':0.99,
            'weight_decay':0.0
            #'grad_clip':clip
        } for l in split_groups], learning_rate=0.0, beta1=0.9,
        beta2=0.99, weight_decay=0.0)
        opt = cls(opt, **kwargs)
        opt.lr, opt.opt_func = listify(lr, layer_groups), opt_func
        print("opt.lr=", opt.lr, "opt.opt_func=", opt.opt_func)
        return opt

    def new(self, layer_groups):
        "Create a new `OptimWrapper` from `self` with another `layer_groups` but the same hyper-parameters."
        opt_func = getattr(self, 'opt_func', self.opt.__class__)
        split_groups = split_bn_bias(layer_groups)
        opt = opt_func([{
            'params': trainable_params(l),
            'learning_rate': 0
        } for l in split_groups])
        return self.create(
            opt_func,
            self.lr,
            layer_groups,
            wd=self.wd,
            true_wd=self.true_wd,
            bn_wd=self.bn_wd)

    def __repr__(self) -> str:
        return f'OptimWrapper over {repr(self.opt)}.\nTrue weight decay: {self.true_wd}'

    #Pytorch optimizer methods
    def step(self) -> None:
        "Set weight decay and step optimizer."
        # weight decay outside of optimizer step (AdamW)
        print("call OptimizerWrapper step..")
        global param_count
        if self.true_wd:
            for lr, wd, pg1, pg2 in zip(self._lr, self._wd,
                                        self.opt._param_groups[::2],
                                        self.opt._param_groups[1::2]):
                print("opt step: wd = ", wd, " lr = ", lr, self._wd, len(pg1['params']))
                for p in pg1['params']:
                    tmp = paddle.full(p.shape, 1-wd*lr)
                    if debug:
                        torch_p = np.load('./weights/before_opt_' + str(param_count) + '.npy')
                        assert np.allclose(torch_p, p.numpy(), atol=1e-5, rtol=1e-5)
                    p.multiply(tmp)

                    if debug:
                        torch_p2 = np.load('./weights/after_opt_' + str(param_count) + '.npy')
                        assert np.allclose(torch_p2, p.numpy(),
                        atol=1e-6, rtol=1e-6)       
                        param_count += 1
                if self.bn_wd:
                    for p in pg2['params']:
                        tmp = paddle.full(p.shape, 1-wd*lr)
                        if debug:
                            torch_p = np.load('./weights/before_opt_' + str(param_count) + '.npy')
                            assert np.allclose(torch_p, p.numpy(), atol=1e-5, rtol=1e-5)
                        p.multiply(tmp)
                        if debug:
                            torch_p2 = np.load('./weights/after_opt_' + str(param_count) + '.npy')
                            assert np.allclose(torch_p2, p.numpy(),
                            atol=1e-5, rtol=1e-5)
                            param_count += 1
            self.set_val('weight_decay', listify(0, self._wd))
        self.opt.step()
        if debug:
            for pg1, pg2 in zip(self.opt._param_groups[::2], self.opt._param_groups[1::2]):
                for p in pg1['params']:
                    weight = np.load('./weights/after_opt2_' + str(param_count) + '.npy')
                    assert np.allclose(weight, p.numpy(), atol=1e-3,
                    rtol=1e-3)
                    param_count += 1
                if self.bn_wd:
                    for p in pg2['params']:
                        weight = np.load('./weights/after_opt2_' + str(param_count) + '.npy')
                        assert np.allclose(weight, p.numpy(),
                        atol=1e-3,
                        rtol=1e-3)
                        param_count += 1

    def zero_grad(self) -> None:
        "Clear optimizer gradients."
        #self.opt.zero_grad()
        self.opt.clear_grad()

    #Passthrough to the inner opt.
    def __getstate__(self):
        print("call __getstate__ ......")
        return self.opt.__getstate__()

    def __setstate__(self, state):
        print("call __setstate__ ......")
        return self.opt.__setstate__(state)

    def state_dict(self):
        print("call state_dict ......")
        return self.opt.state_dict()

    def load_state_dict(self, state_dict):
        print("call load_state_dict ......")
        return self.opt.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        print("call add_param_group......")
        return self.opt.add_param_group(param_group)

    def clear(self):
        "Reset the state of the inner optimizer."
        print("call clear ......")
        sd = self.state_dict()
        sd['state'] = {}
        self.load_state_dict(sd)

    @property
    def param_groups(self):
        print("call param_groups......")
        return self.opt._param_groups

    @property
    def defaults(self):
        print("call defaults......")
        return self.opt.defaults

    @property
    def state(self):
        print("call state......")
        return self.opt.state


    #Hyperparameters as properties
    @property
    def lr(self) -> float:
        print("call lr....")
        #return self._lr[-1]
        return self.opt.get_lr()

    @lr.setter
    def lr(self, val: float) -> None:
        print("call set_lr....")
        self._lr = self.set_val('learning_rate', listify(val, self._lr))
        #self._lr = val
        print('set lr = ', val)
        #self.opt._learning_rate = val
        if isinstance(val, float):
            self.opt.set_lr(val)
        else:
            self.opt.set_lr(val[0])

    @property
    def mom(self) -> float:
        print("call mom....")
        return self._mom[-1]

    @mom.setter
    def mom(self, val: float) -> None:
        print("call set mom....")
        if 'momentum' in self.opt_keys:
            self.set_val('momentum', listify(val, self._mom))
        #elif 'betas' in self.opt_keys:
        #    self.set_val('betas', (listify(val, self._mom), self._beta))
        elif 'beta1' in self.opt_keys:
            self.set_val('beta1', listify(val, self._mom))
        #elif 'beta2' in self.opt_keys:
        #    self.set_val('beta2', listify(val, self._beta))
        self._mom = listify(val, self._mom)
        print('set beta1 = ', val)
        self.opt._beta1=val

    @property
    def beta(self) -> float:
        print("call beta...")
        return None if self._beta is None else self._beta[-1]

    @beta.setter
    def beta(self, val: float) -> None:
        "Set beta (or alpha as makes sense for given optimizer)."
        print("call set beta...")
        if val is None: return
        #if 'betas' in self.opt_keys:
        #    self.set_val('betas', (self._mom, listify(val, self._beta)))
        #if 'beta1' in self.opt_keys:
        #    self.set_val('beta1', listify(val, self._mom))
        if 'beta2' in self.opt_keys:
            self.set_val('beta2', listify(val, self._beta))
        elif 'alpha' in self.opt_keys:
            self.set_val('alpha', listify(val, self._beta))
        self._beta = listify(val, self._beta)
        print('set beta2 = ', val)
        self.opt._beta2=val

    @property
    def wd(self) -> float:
        print("call wd...")
        return self._wd[-1]

    @wd.setter
    def wd(self, val: float) -> None:
        "Set weight decay."
        print("call set wd...")
        if not self.true_wd:
            self.set_val(
                'weight_decay', listify(val, self._wd), bn_groups=self.bn_wd)
        self._wd = listify(val, self._wd)
        print('set wd = ', val)
        #self.opt.regularization=val

    #Helper functions
    def read_defaults(self) -> None:
        "Read the values inside the optimizer for the hyper-parameters."
        self._beta = None
        #if 'lr' in self.opt_keys: self._lr = self.read_val('lr')
        if 'learning_rate' in self.opt_keys: 
            self._lr = self.read_val('learning_rate')
        if 'momentum' in self.opt_keys: self._mom = self.read_val('momentum')
        if 'alpha' in self.opt_keys: self._beta = self.read_val('alpha')
        #if 'betas' in self.opt_keys:
        #    self._mom, self._beta = self.read_val('betas')
        if 'beta1' in self.opt_keys:
            self._mom = self.read_val('beta1')
        if 'beta2' in self.opt_keys:
            self._beta = self.read_val('beta2')
        if 'weight_decay' in self.opt_keys:
            self._wd = self.read_val('weight_decay')
        print("read_defaults:")
        print("lr=", self._lr, "mom=", self._mom, "beta=", self._beta, "wd=", self._wd) 

    def set_val(self, key: str, val, bn_groups: bool = True):
        "Set `val` inside the optimizer dictionary at `key`."
        if is_tuple(val): val = [(v1, v2) for v1, v2 in zip(*val)]
        for v, pg1, pg2 in zip(val, self.opt._param_groups[::2],
                               self.opt._param_groups[1::2]):
            pg1[key] = v
            if bn_groups: pg2[key] = v
        return val

    def read_val(self, key: str):
        "Read a hyperparameter `key` in the optimizer dictionary."
        val = [pg[key] for pg in self.opt._param_groups[::2]]
        if is_tuple(val[0]): val = [o[0] for o in val], [o[1] for o in val]
        return val


class FastAIMixedOptim(OptimWrapper):
    @classmethod
    def create(cls,
               opt_func,
               lr,
               layer_groups,
               model,
               flat_master=False,
               loss_scale=512.0,
               **kwargs):
        "Create an `optim.Optimizer` from `opt_func` with `lr`. Set lr on `layer_groups`."
        opt = OptimWrapper.create(opt_func, lr, layer_groups, **kwargs)
        opt.model_params, opt.master_params = get_master(
            layer_groups, flat_master)
        opt.flat_master = flat_master
        opt.loss_scale = loss_scale
        opt.model = model
        #Changes the optimizer so that the optimization step is done in FP32.
        # opt = self.learn.opt
        mom, wd, beta = opt.mom, opt.wd, opt.beta
        lrs = [lr for lr in opt._lr for _ in range(2)]
        opt_params = [{
            'params': mp,
            'learning_rate': lr
        } for mp, lr in zip(opt.master_params, lrs)]
        opt.opt = opt_func(opt_params)
        opt.mom, opt.wd, opt.beta = mom, wd, beta
        return opt

    def step(self):
        print("call fastai_optim.py step()")
        model_g2master_g(self.model_params, self.master_params,
                         self.flat_master)
        for group in self.master_params:
            for param in group:
                param.grad.div_(self.loss_scale)
        super(FastAIMixedOptim, self).step()
        self.model.zero_grad()
        #Update the params from master to model.
        master2model(self.model_params, self.master_params, self.flat_master)
