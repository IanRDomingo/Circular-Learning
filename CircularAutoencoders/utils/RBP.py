import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import statistics
from tqdm import tqdm
from collections import defaultdict
import pickle

class RBP:
    def __init__(self, model, autoencoding=False, init='random', nofprime=False, skipped=False, device='cpu'):
        self.model = model
        self.autoencoding = autoencoding
        self.skipped = skipped
        self.init = init
        self.nofprime = nofprime
        self.device = device
        if self.init == "random":
            self.rbp_weights = self.init_random_weights(model)
        elif self.init[-2:] == "h5":
            try:
                open(init)
                self.rbp_weights = self.init_rbp_weights_from_array(model, init)
            except:
                raise ValueError("Cannot open the model")
        else:
            raise ValueError("init must be either random or a saved model")
        
    def init_random_weights(self, model):
        rbp_weights = []
        for i, layer in enumerate(list(model.children())):
            layer_type = type(layer).__name__
            if layer_type == "TensorflowConvLayer":
                W = layer.conv.weight
                Wnew = torch.empty_like(W)
                torch.nn.init.xavier_uniform_(Wnew)
                rbp_weights.append(Wnew)

            elif layer_type == "TensorflowLinearLayer":
                W = layer.linear.weight.detach().cpu().numpy()
                b = layer.bias.detach().cpu().numpy()
                if self.skipped:
                    nin, nout = W.shape
                    ll = list(model.children())[-1]
                    if type(ll).__name__ == "TensorflowLinearLayer":
                        nfinal = ll.linear.weight.detach().cpu().numpy()
                    else:
                        raise ValueError("Model contains non-linear layers")
                    Wnew = torch.empty(nin, nfinal).to(self.device)
                    torch.nn.init.xavier_uniform_(Wnew)
                else:
                    nin, nout = W.shape
                    Wnew = torch.empty(nin, nout).to(self.device)
                    torch.nn.init.xavier_uniform_(Wnew)

                rbp_weights.append(Wnew)
            else:
                rbp_weights.append(None)

        return rbp_weights

    def pass_ndedz(self, upper_layer, X, x, ndedz, Wrbp):

        upper_layer_type = type(upper_layer).__name__

        if upper_layer_type == "TensorflowLinearLayer":
            out = torch.tensordot(ndedz, Wrbp, dims=1)
        
        elif upper_layer_type == "TensorflowConvLayer":
            channel_out = ndedz.shape[1]
            channel_in = X.shape[1]
            f_h, f_w = upper_layer.conv.kernel_size
            f_size = [channel_out, channel_in, f_h, f_w]
            pad = (f_size[2] - 1) // 2
            out = self.conv2d_input(X.shape, upper_layer.conv.weight, ndedz, padding=pad)
        
        elif upper_layer_type == "Flatten":
            shape = X.shape
            out = torch.reshape(ndedz, (-1, shape[1], shape[2], shape[3]))



        elif upper_layer_type == "TensorflowMaxPool2d":
            ind = upper_layer.ind
            ks = upper_layer.kernel_size
            st = upper_layer.stride

            out = F.max_unpool2d(ndedz, indices=ind, kernel_size=ks, stride=st)

            

        else:
            raise ValueError("layer not supported")
            
        return out
    
    def conv2d_input(self, input_size, weight, grad_output, stride=1, padding=0, dilation=1, groups=1):
        
        input = grad_output.new_empty(1).expand(input_size)

        return torch.ops.aten.convolution_backward(grad_output, input, weight, None,
                                                _pair(stride), _pair(padding), _pair(dilation),
                                                False, [0], groups, (True, False, False))[0]

    
    def compute_direction(self, Xlist, target):
        ifinal = len(list(self.model.children()))-1
        ndedz = {}
        for i in range(ifinal, -1, -1):
            if i == ifinal:
                if self.autoencoding:
                    ndedz[i] = Xlist[1] - Xlist[-1]
                else:
                    ndedz[i] = target - Xlist[-1]

            else:
                upper_layer = list(self.model.children())[i+1]
                layer = list(self.model.children())[i]
                Wrbp = self.rbp_weights[i+1]
                if self.skipped:
                    if self.nofprime:
                        ndedz[i] = torch.tensordot(ndedz[ifinal], Wrbp.t(), 1)
                    else:
                        ndedz[i] = torch.tensordot(ndedz[ifinal], Wrbp.t(), 1)*self.fprime(layer.activation_fn, Xlist[i+1])
                else:
                    if self.nofprime:
                        ndedz[i] = self.pass_ndedz(upper_layer, Xlist[i+1], Xlist[i+2], ndedz[i+1], Wrbp)
                    else:
                        if type(layer).__name__ in ["TensorflowLinearLayer", "TensorflowConvLayer"]:
                            ndedz[i] = self.pass_ndedz(upper_layer, Xlist[i+1], Xlist[i+2], ndedz[i+1], Wrbp)
                            ndedz[i] = ndedz[i] * self.fprime(layer.activation_fn, Xlist[i+1])
                        else:
                            ndedz[i] = self.pass_ndedz(upper_layer, Xlist[i+1], Xlist[i+2], ndedz[i+1], Wrbp)

        updates = []
    
        for i, l in enumerate(list(self.model.children())):
            if type(l).__name__ == "TensorflowLinearLayer":
                dW, db = self.compute_dense_layer_update_from_ndedz(Xlist[i], ndedz[i])
                updates.append((dW, db))

            elif type(l).__name__ == "TensorflowConvLayer":
                channel_out = ndedz[i].shape[1]
                channel_in = Xlist[i].shape[1]
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
    
    def compute_activities(self, model, input_data, return_all=True):
        Xlist = [input_data]
        for i, l in enumerate(list(model.children())):
            Xlist.append(l(Xlist[-1]))

        if return_all:
            return Xlist
        else:
            return Xlist[-1]    


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
        
    def compute_dense_layer_update_from_ndedz(self, X, ndedz):

        dW = torch.tensordot(X.t(), ndedz, dims=1)
        db = torch.sum(ndedz, dim=0)

        return dW, db
    

    def compute_conv2d_layer_update_from_ndedz(self, input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1):

        weight = grad_output.new_empty(1).expand(weight_size)

        dW = torch.ops.aten.convolution_backward(grad_output, input, weight, None,
                                                _pair(stride), _pair(padding), _pair(dilation),
                                                False, [0], groups, (False, True, False))[1]
        
        db = grad_output.sum(dim=(0, 2, 3))
        
        
        return dW, db
    
    def compute_metrics(self, model, inp):
        if self.autoencoding:
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
        else:
            X, Y = inp[0], inp[1]
            Yhat = model(X.to(self.device)).cpu().detach().numpy()
            X = X.cpu().detach().numpy()
            Y = Y.cpu().detach().numpy()

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
    
    def train_step(self, input_batch, lr, train_bias=True, train_bias_out=True):
        if not self.autoencoding:
            input_x, input_y = input_batch[0], input_batch[1]
            input_x = input_x.to(self.device)
            input_y = input_y.to(self.device)
            Xlist = self.compute_activities(self.model, input_x, return_all=True)
            direction = self.compute_direction(Xlist, target=input_y)
            batchsize = input_x.shape[0]
        else:
            input_x = input_batch
            Xlist = self.compute_activities(self.model, input_x, return_all=True)
            direction = self.compute_direction(Xlist, target=None)
            batchsize = input_batch.shape[0]

        for i, layer in enumerate(list(self.model.children())):
            layer_type = type(layer).__name__
            if not layer_type in ["TensorflowLinearLayer", "TensorflowConvLayer"]:
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
                
                if train_bias or (train_bias_out and i == len(self.model.layers)-1):
                    new_weights = [new_W, new_b]
                    layer.set_weights(new_weights, self.device)
                else:
                    new_weights = [new_W, b]
                    layer.set_weights(new_weights, self.device)

    def decay_learning_rate(self, initial_lr, decay_rate, step):
        lr = initial_lr*(1. / (1. + decay_rate * step))
        return lr
    
    def train(self, dataset, lr=.01, momentum=.8, l2=.0, constraint=None, epochs=50, valid_dataset=None, train_bias=True, train_bias_out=True, verbose=True, cbks=[], decay_lr=True):
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
                if self.autoencoding:
                    train_batch = train_batch.to(self.device)
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
                    vals = self.compute_metrics(self.model, train_batch)
                    for q, w in vals.items():
                        moving_metrics[q].append(float(w))

            self.trainAcc[e] = self.test(self.model, dataset=dataset)

            if valid_dataset is not None:
                val_metrics = self.test(self.model, dataset=valid_dataset)
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
    
    def test(self, model, dataset, prefix = 'val_'):
        batch_history = defaultdict(list)
        for batch in dataset:
            if self.autoencoding:
                batch = batch.to(self.device)
            vals = self.compute_metrics(model, batch)
            for k, v in vals.items():
                batch_history[prefix+k].append(v)

        rval = {}

        for k,v in batch_history.items():
            rval[k] = np.mean(batch_history[k])
        return rval
    