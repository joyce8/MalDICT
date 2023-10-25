# Adapted from https://github.com/NeuromorphicComputationResearchProgram/MalConv2/blob/main/LowMemConv.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_zeros_hook(module, grad_input, grad_out):
    """
    This function is used to replace gradients that are all zeros with None in
    pytorch None will not get back-propogated. So we use this as an
    approximation to sparse BP to avoid redundant and useless work.
    """
    grads = []
    with torch.no_grad():
        for g in grad_input:
            if torch.nonzero(g).shape[0] == 0:
                grads.append(g.to_sparse())
            else:
                grads.append(g)
                
    return tuple(grads)


class CatMod(torch.nn.Module):
    def __init__(self):
        super(CatMod, self).__init__()

    def forward(self, x):
        return torch.cat(x, dim=2)
    

class LowMemConvBase(nn.Module):
    
    def __init__(self, chunk_size=65536, overlap=512, min_chunk_size=1024):
        """
        chunk_size - How many bytes at a time to process. Increasing may
                     improve compute efficent, but use more memory. Total
                     memory use will be a function of chunk_size, and not of
                     the length of the input sequence L
        overlap - How many bytes of overlap to use between chunks
        """

        super(LowMemConvBase, self).__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.cat = CatMod()
        self.cat.register_backward_hook(drop_zeros_hook)
        self.receptive_field = None
        self.dummy_tensor = torch.ones(1, dtype=torch.float32,
                                       requires_grad=True)


    def determinRF(self):
        """
        Let's determine the receptive field & stride of our sub-network.
        """
        
        if self.receptive_field is not None:
            return self.receptive_field, self.stride, self.out_channels
        cur_device = next(self.embd.parameters()).device
        min_rf = 1
        max_rf = self.chunk_size
        with torch.no_grad():
            tmp = torch.zeros((1,max_rf)).long().to(cur_device)
            while True:
                test_size = (min_rf+max_rf)//2
                is_valid = True
                try:
                    self.processRange(tmp[:,0:test_size])
                except:
                    is_valid = False
                
                if is_valid:
                    max_rf = test_size
                else:
                    min_rf = test_size+1
                if max_rf == min_rf:
                    self.receptive_field = min_rf
                    out_shape = self.processRange(tmp).shape
                    self.stride = self.chunk_size//out_shape[2]
                    self.out_channels = out_shape[1]
                    break

        return self.receptive_field, self.stride, self.out_channels


    def pool_group(self, *args):
        x = self.cat(args)
        x = self.pooling(x)
        return x


    def seq2fix(self, x, pr_args={}):
        """
        Takes in an input LongTensor of (B, L) that will be converted to a
        fixed length representation (B, C), where C is the number of channels
        provided by the base_network  given at construction. 
        """

        receptive_window, stride, out_channels = self.determinRF()
        if x.shape[1] < receptive_window:
            x = F.pad(x, (0, receptive_window-x.shape[1]), value=0)
        batch_size = x.shape[0]
        length = x.shape[1]
        winner_values = np.zeros((batch_size, out_channels))-1.0
        winner_indices = np.zeros((batch_size, out_channels), dtype=np.int64)
        cur_device = next(self.embd.parameters()).device
        step = self.chunk_size
        start = 0
        end = start+step
        with torch.no_grad():
            while (start < end and
                   (end-start) >= max(self.min_chunk_size, receptive_window)):
                x_sub = x[:,start:end]
                x_sub = x_sub.to(cur_device)
                activs = self.processRange(x_sub.long(), **pr_args)
                k_size = activs.shape[2]
                activ_win, activ_indx = F.max_pool1d(activs,
                                                     kernel_size=k_size,
                                                     return_indices=True)
                activ_win = activ_win.cpu().numpy()[:,:,0]
                activ_indx = activ_indx.cpu().numpy()[:,:,0]
                selected = winner_values < activ_win
                winner_indices[selected] = activ_indx[selected]*stride + start 
                winner_values[selected]  = activ_win[selected]
                start = end
                end = min(start+step, length)

        final_indices = [np.unique(winner_indices[b,:])
                         for b in range(batch_size)]
        chunk_list = [[x[b:b+1,max(i-receptive_window,0):
                         min(i+receptive_window,length)]
                       for i in final_indices[b]] for b in range(batch_size)]
        chunk_list = [torch.cat(c, dim=1)[0,:] for c in chunk_list]
        x_selected = torch.nn.utils.rnn.pad_sequence(chunk_list,
                                                     batch_first=True)
        x_selected = x_selected.to(cur_device)
        x_selected = self.processRange(x_selected.long(), **pr_args)
        x_selected = self.pooling(x_selected)
        x_selected = x_selected.view(x_selected.size(0), -1)

        return x_selected
