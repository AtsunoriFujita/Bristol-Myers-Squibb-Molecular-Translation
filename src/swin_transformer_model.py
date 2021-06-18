import math
import numpy as np
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from fairseq.models import *
from fairseq.modules import *
from swin_transformer import *

# https://arxiv.org/pdf/1411.4555.pdf
# 'Show and Tell: A Neural Image Caption Generator' - Oriol Vinyals, cvpr-2015

STOI = {
    '<sos>': 190,
    '<eos>': 191,
    '<pad>': 192,
}

image_size = 384
vocab_size = 193
max_length = 280

image_dim = 1024
text_dim = 1024
decoder_dim = 1024
num_layer = 6
num_head = 8
ff_dim = 2048


class CNN(nn.Module):
    def __init__(self,):
        super(CNN, self).__init__()

        self.e = swin_base_patch4_window12_384_in22k(pretrained=True)
        for p in self.e.parameters():
            p.requires_grad = True#False

    def forward(self, image):    
        x = self.e.forward_features(image)

        return x


#https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

# ------------------------------------------------------
# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
# https://stackoverflow.com/questions/46452020/sinusoidal-embedding-attention-is-all-you-need

class PositionEncode1D(nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        assert (dim % 2 == 0)
        self.max_length = max_length

        d = torch.exp(torch.arange(0., dim, 2) * (-math.log(10000.0) / dim))
        position = torch.arange(0., max_length).unsqueeze(1)
        pos = torch.zeros(1, max_length, dim)
        #pos.require_grad = False
        pos[0, :, 0::2] = torch.sin(position * d)
        pos[0, :, 1::2] = torch.cos(position * d)
        self.register_buffer('pos', pos)

    def forward(self, x):
        batch_size, T, dim = x.shape
        x = x + self.pos[:,:T]
        return x

# https://gitlab.maastrichtuniversity.nl/dsri-examples/dsri-pytorch-workspace/-/blob/c8a88cdeb8e1a0f3a2ccd3c6119f43743cbb01e9/examples/transformer/fairseq/models/transformer.py
#https://github.com/pytorch/fairseq/issues/568
# fairseq/fairseq/models/fairseq_encoder.py

# https://github.com/pytorch/fairseq/blob/master/fairseq/modules/transformer_layer.py
class TransformerEncode(FairseqEncoder):

    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__({})
        #print('my TransformerEncode()')

        self.layer = nn.ModuleList([
            TransformerEncoderLayer(Namespace({
                'encoder_embed_dim': dim,
                'encoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'encoder_normalize_before': True,
                'encoder_ffn_embed_dim': ff_dim,
            })) for i in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):# T x B x C
        #print('my TransformerEncode forward()')
        for layer in self.layer:
            x = layer(x)
        x = self.layer_norm(x)
        return x


# https://fairseq.readthedocs.io/en/latest/tutorial_simple_lstm.html
# see https://gitlab.maastrichtuniversity.nl/dsri-examples/dsri-pytorch-workspace/-/blob/c8a88cdeb8e1a0f3a2ccd3c6119f43743cbb01e9/examples/transformer/fairseq/models/transformer.py
class TransformerDecode(FairseqIncrementalDecoder):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__({})
        #print('my TransformerDecode()')

        self.layer = nn.ModuleList([
            TransformerDecoderLayer(Namespace({
                'decoder_embed_dim': dim,
                'decoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'decoder_normalize_before': True,
                'decoder_ffn_embed_dim': ff_dim,
                #'decoder_learned_pos': True,
                #'cross_self_attention': True,
                'activation-fn': 'gelu',
            })) for i in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)


    def forward(self, x, mem, x_mask):
            #print('my TransformerDecode forward()')
            for layer in self.layer:
                x = layer(x, mem, self_attn_mask=x_mask)[0]
            x = self.layer_norm(x)
            return x  # T x B x C

    #def forward_one(self, x, mem, incremental_state):
    def forward_one(self,
            x   : torch.Tensor,
            mem : torch.Tensor,
            incremental_state : Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]
    )-> torch.Tensor:
        x = x[-1:]
        for layer in self.layer:
            x = layer(x, mem, incremental_state=incremental_state)[0]
        x = self.layer_norm(x)
        return x


class Net(nn.Module):

    def __init__(self, ):
        super(Net, self).__init__()
        self.cnn = CNN()
        self.image_encode = nn.Identity()

        # ---
        self.text_pos = PositionEncode1D(text_dim, max_length)
        self.token_embed = nn.Embedding(vocab_size, text_dim)
        self.text_decode = TransformerDecode(decoder_dim, ff_dim, num_head, num_layer)

        # ---
        self.logit = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.1)

        # ----
        # initialization
        self.token_embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-0.1, 0.1)

    @torch.jit.unused
    def forward(self, image, token, length):
        device = image.device
        batch_size = len(image)
        # ---
        image_embed = self.cnn(image)
        image_embed = self.image_encode(image_embed).permute(1, 0, 2).contiguous()
        
        # Noize Injection
        if self.training:
            probs = torch.empty(token.size()).uniform_(0, 1).to(device)
            probs = torch.where(token < 190, probs, torch.empty(token.size(), dtype=torch.float).fill_(1).to(device))
            token = torch.where(probs > 0.15, token, torch.randint(0, 190, token.size(), dtype=torch.int64).to(device))
        
        text_embed = self.token_embed(token)
        text_embed = self.text_pos(text_embed).permute(1, 0, 2).contiguous()
        
        text_mask = np.triu(np.ones((max_length, max_length)), k=1).astype(np.uint8)
        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask) == 1).to(device)

        x = self.text_decode(text_embed, image_embed, text_mask)
        x = x.permute(1, 0, 2).contiguous()
        x = self.dropout(x)

        logit = self.logit(x) 
        return logit

    @torch.jit.export
    def forward_argmax_decode(self, image):

        image_dim = 1024
        text_dim = 1024
        decoder_dim = 1024
        num_layer = 6 #3
        num_head = 8
        ff_dim = 2048

        STOI = {
            '<sos>': 190,
            '<eos>': 191,
            '<pad>': 192,
        }

        image_size = 384
        vocab_size = 193
        max_length = 280

        # ---------------------------------
        device = image.device
        batch_size = len(image)

        image_embed = self.cnn(image)
        image_embed = self.image_encode(image_embed).permute(1, 0, 2).contiguous()

        token = torch.full((batch_size, max_length), STOI['<pad>'], dtype=torch.long, device=device)
        text_pos = self.text_pos.pos
        token[:, 0] = STOI['<sos>']

        # -------------------------------------
        eos = STOI['<eos>']
        pad = STOI['<pad>']

        # https://github.com/alexmt-scale/causal-transformer-decoder/blob/master/tests/test_consistency.py
        # slow version
        #if 1:
        #    for t in range(max_length-1):
        #        last_token = token [:,:(t+1)]
        #        text_embed = self.token_embed(last_token)
        #        text_embed = self.text_pos(text_embed).permute(1,0,2).contiguous() #text_embed + text_pos[:,:(t+1)] #
        
        #        text_mask = np.triu(np.ones((t+1, t+1)), k=1).astype(np.uint8)
        #        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask)==1).to(device)
        
        #        x = self.text_decode(text_embed, image_embed, text_mask)
        #        x = x.permute(1,0,2).contiguous()
        
        #        l = self.logit(x[:,-1])
        #        k = torch.argmax(l, -1)  # predict max
        #        token[:, t+1] = k
        #        if ((k == eos) | (k == pad)).all(): break

        # fast version
        if 1:
            # incremental_state = {}
            incremental_state = torch.jit.annotate(
                Dict[str, Dict[str, Optional[torch.Tensor]]],
                torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {}),
            )
            for t in range(max_length - 1):
                # last_token = token [:,:(t+1)]
                # text_embed = self.token_embed(last_token)
                # text_embed = self.text_pos(text_embed) #text_embed + text_pos[:,:(t+1)] #

                last_token = token[:, t]
                text_embed = self.token_embed(last_token)
                text_embed = text_embed + text_pos[:, t]  #
                text_embed = text_embed.reshape(1, batch_size, text_dim)

                x = self.text_decode.forward_one(text_embed, image_embed, incremental_state)
                x = x.reshape(batch_size, decoder_dim)

                l = self.logit(x)
                k = torch.argmax(l, -1)  # predict max
                token[:, t + 1] = k
                if ((k == eos) | (k == pad)).all():
                    break

        predict = token[:, 1:]
        return predict


# loss #################################################################
def seq_cross_entropy_loss(logit, token, length):
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    truth = pack_padded_sequence(truth, L, batch_first=True).data
    loss = F.cross_entropy(logit, truth, ignore_index=STOI['<pad>'])
    return loss


# https://www.aclweb.org/anthology/2020.findings-emnlp.276.pdf
def seq_anti_focal_cross_entropy_loss(logit, token, length):
    gamma = 1.0 # {0.5,1.0}
    label_smooth = 0.90

    #---
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    truth = pack_padded_sequence(truth, L, batch_first=True).data
    #loss = F.cross_entropy(logit, truth, ignore_index=STOI['<pad>'])
    #non_pad = torch.where(truth != STOI['<pad>'])[0]  # & (t!=STOI['<sos>'])

    # ---
    #p = F.softmax(logit,-1)
    #logp = - torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))

    logp = F.log_softmax(logit, -1)
    logp = logp.gather(1, truth.reshape(-1,1)).reshape(-1)
    p = logp.exp()

    #loss = - ((1 - p) ** gamma)*logp  #focal
    loss = - ((1 + p) ** gamma)*logp  #anti-focal
    loss = loss.mean()
    return loss


def np_loss_cross_entropy(probability, truth):
    batch_size = len(probability)
    truth = truth.reshape(-1)
    p = probability[np.arange(batch_size),truth]
    loss = -np.log(np.clip(p, 1e-6, 1))
    loss = loss.mean()
    return loss