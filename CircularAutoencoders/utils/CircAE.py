import numpy as np
from collections import defaultdict
from tqdm import tqdm
import statistics
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import sys
import pickle

class CircularAE:
    def __init__(self, model, ncirc, combine='PR', combine_lambda=0.0, no_target_flag=True, t2_minus_t1=False, nofprime=False, device = 'cpu'):
        self.model = model
        self.ncirc = ncirc
        self.combine = combine
        self.combine_lambda = combine_lambda
        self.no_target_flag = no_target_flag
        self.t2_minus_t1 = t2_minus_t1
        self.nofprime = nofprime
        self.device = device
    
    def train_step(self, input_batch, lr, train_bias=True, train_bias_out=True):

        Xlist = self.compute_activities(self.model, input_batch, return_all=True)

        direction = self.compute_direction(Xlist)
        
        batchsize = input_batch.shape[0]

        for i, layer in enumerate(list(self.model.children())):
            layer_type = type(layer).__name__ # it can be one of the followings: dense, conv2d, max, up, input
            if layer_type not in ["TensorflowLinearLayer", "TensorflowConvLayer"]: # layers with trainable parameters
                continue
            else:
                
                dW, db = direction[i]
                dW = dW.detach().cpu().numpy()
                db = db.detach().cpu().numpy()

                if layer_type == "TensorflowConvLayer":
                    W = layer.conv.weight.detach().cpu().numpy()
                    b = layer.bias.detach().cpu().numpy()
                elif layer_type == "TensorflowLinearLayer":
                    W = layer.linear.weight.detach().cpu().numpy()
                    b = layer.bias.detach().cpu().numpy()
                    dW = dW.T

                
                if self.l2 is not None and self.l2 > 0.0:
                    dW = dW - self.l2 * W
                    db = db - self.l2 * b

                if self.momentum > 0.0:    
                    dW_prev, db_prev = self.layers_momentum[i]

                    dW = (self.momentum * dW_prev) + ((1. - self.momentum) * dW)
                    db = (self.momentum * db_prev) + ((1. - self.momentum) * db)

                    self.layers_momentum[i] = (dW, db)

                if type(lr) == float:
                    new_W = W + lr*dW/batchsize
                    new_b = b + lr*db/batchsize
                else:
                    assert type(lr) == list, type(lr)
                    new_W = W + lr[i]*dW/batchsize
                    new_b = b + lr[i]*db/batchsize

                if self.constraint is not None:
                    new_W = self.constraint(new_W)
                    new_b = self.constraint(new_b)

                
                if train_bias or (train_bias_out and i == len(self.model.layers)-1):
                    new_weights = [new_W, new_b]
                    layer.set_weights(new_weights, self.device)
                else:
                    new_weights = [new_W, b]
                    layer.set_weights(new_weights, self.device)
    
    def train(self, dataset, lr=0.01, momentum=0.8, l2=0.0, constraint=None, epochs=50, 
              valid_dataset=None, train_bias=True, train_bias_out=True, verbose=True, 
              cbks=[], decay_lr=True, predecessors = []):
        
        self.initial_lr = lr
        self.momentum = momentum
        self.l2 = l2
        self.constraint = constraint
        self.trainAcc = {}
        self.testAcc = {}


        if self.momentum > 0.0:
            self.layers_momentum = {}
            for i, layer in enumerate(self.model.children()):
                layer_type = type(layer).__name__

                if layer_type == "TensorflowConvLayer":
                    W = layer.conv.weight.detach().cpu().numpy()
                    b = layer.bias.detach().cpu().numpy()
                    dW = np.zeros_like(W, dtype=np.float32)
                    db = np.zeros_like(b, dtype=np.float32)
                elif layer_type == "TensorflowLinearLayer":
                    W = layer.linear.weight.detach().cpu().numpy()
                    b = layer.bias.detach().cpu().numpy()
                    dW = np.zeros_like(W, dtype=np.float32)
                    db = np.zeros_like(b, dtype=np.float32)
                else:
                    dW, db = (None, None)
                self.layers_momentum[i] = (dW, db)

        history = defaultdict(list)
        
        if type(self.initial_lr) == float:
            decay_rate = self.initial_lr/epochs
        else:
            decay_rate = [j/epochs for j in self.initial_lr]
        total_steps = 0
        for e in range(epochs):
            metrics = {}
            moving_metrics = defaultdict(list)
            step = 0
            for train_batch in tqdm(dataset):
                train_batch = train_batch.to(self.device)
                for predecessor in predecessors:
                    train_batch = predecessor(train_batch)
                if decay_lr:
                    if type(self.initial_lr) == float:
                        lr = self.decay_learning_rate(self.initial_lr, decay_rate, total_steps)
                    else:
                        lr = [self.decay_learning_rate(j, v, total_steps) for (j,v) in zip(self.initial_lr, decay_rate)]
                else:
                    lr = self.initial_lr
                self.train_step(train_batch, lr)
                step += 1        
                total_steps += 1
                if step % 200 == 0:
                    vals = self.compute_metrics(self.model, train_batch, predecessors=predecessors)
                    for q, w in vals.items():
                        moving_metrics[q].append(float(w))            
            
            self.trainAcc[e] = self.test(self.model, dataset=dataset, predecessors=predecessors)

            if valid_dataset is not None:
                val_metrics = self.test(self.model, dataset=valid_dataset, predecessors=predecessors)
                self.testAcc[e] = val_metrics
                metrics.update(val_metrics)
            
            for r,s in moving_metrics.items():
                metrics[r] = statistics.mean(s)
            for k,v in metrics.items():
                history[k].append(v)
            
            # Print status.
            if verbose:
                self.print_metrics(history)

        return self.model, history

    def compute_activities(self, model, input_data, return_all=True):
        Xlist = [input_data]
        for i, l in enumerate(list(model.children())):
            Xlist.append(l(Xlist[-1]))

        if return_all:
            return Xlist
        else:
            return Xlist[-1]    

    def decay_learning_rate(self, initial_lr, decay_rate, step):
        lr = initial_lr*(1. / (1. + decay_rate * step))
        return lr
    
    def compute_direction(self, Xlist):
        Xlist2 = Xlist
        for i in range(self.ncirc):
            Xlist2  = self.recirculate(Xlist2, return_all=True)
        updates = self.compute_updates_from_Xlists(Xlist, Xlist2)
        return updates
    
    def compute_updates_from_Xlists(self, Xlist, Xlist2):
        ndedz = {}
        for i, l in enumerate(list(self.model.children())):
            if i == len(list(self.model.children()))-1:

                if self.no_target_flag:
                    ndedz[i] = Xlist2[i+1] - Xlist[i+1]
                else:
                    ndedz[i] = Xlist[1] - Xlist2[1]
            else:
                if self.t2_minus_t1:
                    ndedz[i] = (Xlist2[i+1] - Xlist[i+1])
                else:
                    ndedz[i] = (Xlist[i+1] - Xlist2[i+1])

        updates = []

        for i, l in enumerate(list(self.model.children())):
            layer_type = type(l).__name__
            if layer_type == "TensorflowLinearLayer":
                if i == len(list(self.model.children()))-1:
                    pass
                else:
                    if not self.nofprime:
                        ndedz[i] = ndedz[i] * self.fprime(l.activation_fn, Xlist[i+1])
                    else:
                        pass
                dW,db = self.compute_dense_layer_update_from_ndedz(Xlist[i], ndedz[i])
                updates.append((dW,db))

            elif layer_type == "TensorflowConvLayer":

                if not self.nofprime:
                    ndedz[i] = ndedz[i] * self.fprime(l.activation_fn, Xlist[i+1])

                channel_out = ndedz[i].shape[1]
                channel_in = Xlist2[i].shape[1]
                f_h, f_w = l.conv.kernel_size
                f_size = [channel_out, channel_in, f_h, f_w]
                
                s = [1, 1, 1, 1]

                p = l.conv.padding.upper()

                pad = (f_size[2] - 1) // 2
                
                dW,db = self.compute_conv2d_layer_update_from_ndedz(Xlist[i], f_size, ndedz[i], padding=pad)
            
                updates.append((dW,db))
                
            else:
                updates.append(())

        return updates
    

    def compute_conv2d_layer_update_from_ndedz(self, input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1):

        weight = grad_output.new_empty(1).expand(weight_size)

        dW = torch.ops.aten.convolution_backward(grad_output, input, weight, None,
                                                _pair(stride), _pair(padding), _pair(dilation),
                                                False, [0], groups, (False, True, False))[1]
        
        db = grad_output.sum(dim=(0, 2, 3))
        return dW, db
    
    def compute_dense_layer_update_from_ndedz(self, X, ndedz):
        dW = torch.tensordot(X.t(), ndedz, dims=1)
        db = torch.sum(ndedz, dim=0)

        return dW, db
    
    def recirculate(self, Xlist, return_all=False):
        if self.combine == 'PR':
            rval = [Xlist[-1]]
            for i, l in enumerate(list(self.model.children())): 
                rval.append(l(rval[-1]))
        else:
            raise IOError('Unknown combination algorithm {}.'.format(self.combine))
        if return_all:
            return rval
        else:
            return rval[-1]
        
    def fprime(self, layer, x):
        if isinstance(layer, torch.nn.Tanh):
            return (1. - torch.square(x))
        elif isinstance(layer, torch.nn.Sigmoid):
            return x * (1. - x)
        elif isinstance(layer, torch.nn.Identity):
            return torch.ones_like(x, dtype=torch.float32)
        elif isinstance(layer, torch.nn.ReLU):
            return torch.where(x > 0., torch.ones_like(x), torch.zeros_like(x))
        elif isinstance(layer, torch.nn.SELU):
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            return torch.where(x > 0., scale * torch.ones_like(x), x + (alpha*scale))
        elif layer is None:
            return x
        else:
            raise Exception('Unrecognized transfer function.')
        
    def compute_metrics(self, model, inp, predecessors):
        Y = inp.cpu().detach().numpy()
        Yhat = model(inp).cpu().detach().numpy()
        metrics = {}
        metrics['mse'] = np.mean((Yhat - Y) ** 2)
        metrics['mae'] = np.mean(np.abs(Yhat - Y))
        last_trainable_layer = list(self.model.children())[-1].activation_fn

        if type(last_trainable_layer).__name__ == 'Sigmoid':

            epsilon = 1e-15
            Yhat = np.clip(Yhat, epsilon, 1. - epsilon)
            bce = -np.mean(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))

            metrics['kl_loss'] = np.mean(bce)
            metrics['acc'] = np.mean((Y > 0.5) == (Yhat > 0.5))
        elif type(last_trainable_layer).__name__ == 'Identity':
            metrics['loss'] = metrics['mse']
        else:
            raise Exception('unrecognized output type.')
        return metrics

    @staticmethod
    def print_metrics(history):
        #print(torch.cuda.memory_summary())
        #objgraph.show_most_common_types()
        assert isinstance(history, (defaultdict, dict))
        if len(history) == 0:
            return
        nepochs = len(history['mse'])
        print('Epoch: {}  '.format(nepochs-1) + '  '.join(['{}: {}'.format(k,v[-1]) for k,v in history.items()]))
        print('========================')
        return
    
    def test(self, model, dataset, prefix = 'val_', predecessors = []):
        batch_history = defaultdict(list)
        for batch in dataset:
            nbatch = batch.to(self.device)
            for predecessor in predecessors:
                nbatch = predecessor(nbatch)
            vals = self.compute_metrics(model, nbatch, predecessors)
            for k, v in vals.items():
                batch_history[prefix+k].append(v)

        rval = {}

        for k,v in batch_history.items():
            rval[k] = np.mean(batch_history[k])
        return rval
    

    def asynchronousTrainSetup(self, lr=0.01, momentum=0.8, l2=0.0, constraint=None):
        
        self.initial_lr = lr
        self.momentum = momentum
        self.l2 = l2
        self.constraint = constraint

        if self.momentum > 0.0:
            self.layers_momentum = {}
            for i, layer in enumerate(self.model.children()):
                layer_type = type(layer).__name__

                if layer_type == "TensorflowConvLayer":
                    W = layer.conv.weight.detach().cpu().numpy()
                    b = layer.bias.detach().cpu().numpy()
                    dW = np.zeros_like(W, dtype=np.float32)
                    db = np.zeros_like(b, dtype=np.float32)
                elif layer_type == "TensorflowLinearLayer":
                    W = layer.linear.weight.detach().cpu().numpy()
                    b = layer.bias.detach().cpu().numpy()
                    dW = np.zeros_like(W, dtype=np.float32)
                    db = np.zeros_like(b, dtype=np.float32)
                else:
                    dW, db = (None, None)
                self.layers_momentum[i] = (dW, db)