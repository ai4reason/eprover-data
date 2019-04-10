import os
import glob
import pickle
import gzip
from tqdm import tqdm
import concurrent.futures
import numpy as np
import torch
import torch.nn as nn
import argparse

import rnnennigma

import sys
sys.setrecursionlimit(10000)


parser = argparse.ArgumentParser(description = ("Export model"), formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('model',
                    type=str,
                    help='input model filename')

parser.add_argument('out_dir',
                    type=str,
                    help='output directory')

args = parser.parse_args()

log_filename = '.'.join(args.model.split('.')[:-1])

with open(log_filename, 'r') as f:
    for line in f:
        l = line.split(' ; ')

        if len(l) > 3 and l[3] == 'data':
            datafilename = l[4][:-1]
            break

print('Data filename:', datafilename)

data = pickle.load(open(datafilename, 'rb'))

filename = args.model

model_params = pickle.load(open(filename + '.model_args', 'rb'))
new_model_params = model_params[1]
new_model_params['device'] = 'cpu'

#new_model_params['random_order']
#new_model_params['random_order'] = False

model = rnnennigma.RNNennigma(new_model_params).cpu()
model.load_state_dict(torch.load(filename + '.model_parameters', map_location='cpu'))
model.eval()

results = dict()

for s, i in data['str_to_int'].items():
    #s_fnc = lambda x: model.symbol_name(s, x)
    
    if i in data['arity']:
        results[s] = torch.jit.trace(model.symbol_value(i), torch.rand(model.dim * data['arity'][i]))
    else:
        #results[s] = torch.jit.torch.tensor(s_fnc([]).clone().detach())
        pass

if 'conjecture_dim' in new_model_params:
    in_final = torch.rand(model.conjecture_dim + model.dim)
else:
    in_final = torch.rand(2 * model.dim)

traced_final = torch.jit.trace(model.final, in_final)
#trace_clause_net = torch.jit.trace(model.clause_net, torch.rand((10, 1, model.dim)))
traced_s_emb = torch.jit.trace(model.s_emb, torch.LongTensor([1]))

if args.out_dir.endswith('/'):
    new_dir = args.out_dir
else:
    new_dir = args.out_dir + '/'
    
new_dir_str = new_dir + 'str/'

os.mkdir(new_dir)
#new_dir_int = new_dir + 'int/'
os.mkdir(new_dir_str)
#os.mkdir(new_dir_int)

traced_final.save(new_dir + 'final.pt')
traced_s_emb.save(new_dir + 's_emb.pt')

load_trace_s_emb = torch.jit.load(new_dir + 's_emb.pt')
load_trace_final = torch.jit.load(new_dir + 'final.pt')

with open(new_dir + 'translate_const.txt', 'w') as f_const:
    #with open(new_dir + 'translate_fnc_pred.text', 'w') as f_fnc_pred: 
        for s, i in data['str_to_int'].items():
        
            if i not in data['arity']:      
                f_const.write(s + ' ' + str(model.uncommon_val(i)) + '\n')
            else:
                #f_fnc_pred.write(s + ' ' + str(i) + '\n')
                results[s].save(new_dir_str + s + '.pt')
                #results[s].save(new_dir_int + str(i) + '.pt')

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, model.dim)

def postprocess(x):
    return x[0][-1][0]

clausenet = nn.Sequential(Lambda(preprocess), model.clause_net, Lambda(postprocess))

rnd_in = torch.rand(10 * model.dim)

traced_clausenet = torch.jit.trace(clausenet, rnd_in)

traced_clausenet.save(new_dir + 'clausenet.pt')
clausenet_load = torch.jit.load(new_dir + 'clausenet.pt')


conjecturenet = nn.Sequential(Lambda(preprocess), model.conjecture_net, Lambda(postprocess))

traced_conjecturenet = torch.jit.trace(conjecturenet, rnd_in)

traced_conjecturenet.save(new_dir + 'conjecturenet.pt')

conjecturenet_load = torch.jit.load(new_dir + 'conjecturenet.pt')

with open(new_dir + 'model.info', 'w') as f:
    f.write(args.model)
