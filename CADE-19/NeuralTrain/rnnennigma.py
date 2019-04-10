import torch
import torch.nn as nn
import random
from tqdm import tqdm
#import torch.multiprocessing as mp
#from numba import njit, jit

random.seed(12345)

class RNNennigma(nn.Module):
##class RNNennigma(torch.jit.ScriptModule):
    ###__constants__ = ['dim',
    ###                 'clause_bidir']
    
    def __init__(self, model_params):
        super(RNNennigma, self).__init__()

        self.device = model_params['device']

        self.dim = model_params['dimension']

        if 'comb_type' in model_params:
            self.comb_type = model_params['comb_type']
        else:
            self.comb_type = 'LinearReLU'

        if 'final_type' in model_params:
            self.final_type = model_params['final_type']
        else:
            self.final_type = 'LinearReLU'

        self.eval_mode = 'eval_mode' in model_params
            
        self.int_to_str = model_params['int_to_str']
        self.str_to_int = model_params['str_to_int']
        self.symbols = model_params['symbols']
        self.train_symbols = model_params['symbols']
        self.arity = model_params['arity']
        self.symbol_to_type = model_params['symbol_to_type']
        self.uncommon_symbols_repr = model_params['uncommon_symbols_repr']
        self.uncommon_trans = model_params['uncommon_trans']

        self.random_order = model_params['random_order']
        self.equality_symb = self.str_to_int['=']
        
        self.pred_depth = model_params['pred_depth']
        self.fnc_depth = model_params['fnc_depth']
        self.clause_depth = model_params['clause_depth']

        self.use_conjecture = model_params['conjecture']

        if model_params['nthreads'] == 1:
            self.parallel = False
        else:
            self.parallel = True

            # try:
            #     mp.set_start_method('spawn')
            # except RuntimeError:
            #     pass
            
            self.nthreads = model_params['nthreads']
            
        if self.use_conjecture:
            self.conjecture_depth = model_params['conjecture_depth']        
            self.cache_conjecture = dict()
            self.fileid_to_conj = model_params['fileid_to_conj']
            self.all_files_from_id = model_params['all_files_from_id']

            if 'conjecture_dim' in model_params:
                if model_params['conjecture_dim']:
                    self.conjecture_dim = model_params['conjecture_dim']
                else:
                    self.conjecture_dim = self.dim
            else:
                self.conjecture_dim = self.dim
        else:
            self.conjecture_dim = 0
            
        self.all_types = model_params['all_types']

        self.symbol = nn.ModuleDict()

        self.cache = model_params['cache']
        self.cache_repr = dict()


        self.s_emb = nn.Embedding(len(self.str_to_int), self.dim).to(self.device)

        #self.long_tensor = {i: torch.LongTensor([i]).to(self.device) for i in self.int_to_str}
        
        def new_comb(s):
            s_arity = self.arity[s]
            assert s_arity > 0

            if self.symbol_to_type[s] in {'FUNCTION', 'SKOLEM_FUNCTION'}:
                comb_depth = self.fnc_depth
            elif self.symbol_to_type[s] in {'PREDICATE', 'SKOLEM_PREDICATE'}:
                comb_depth = self.pred_depth
            elif {x for x in self.int_to_str[s]} <= {'~', '='}:
                comb_depth = self.pred_depth
            else:
                raise

            if self.comb_type == 'LinearReLU':
                comb_str = 'nn.Sequential(' + ','.join((comb_depth - 1) * ['nn.Linear(self.dim * s_arity, self.dim * s_arity), nn.ReLU()'] + ['nn.Linear(self.dim * s_arity, self.dim), nn.ReLU()']) + ').to(self.device)'
            elif self.comb_type == 'LinearReLU6':
                comb_str = 'nn.Sequential(' + ','.join((comb_depth - 1) * ['nn.Linear(self.dim * s_arity, self.dim * s_arity), nn.ReLU()'] + ['nn.Linear(self.dim * s_arity, self.dim), nn.ReLU6()']) + ').to(self.device)'
            elif self.comb_type == 'LinearTanh':
                comb_str = 'nn.Sequential(' + ','.join((comb_depth - 1) * ['nn.Linear(self.dim * s_arity, self.dim * s_arity), nn.Tanh()'] + ['nn.Linear(self.dim * s_arity, self.dim), nn.Tanh()']) + ').to(self.device)'
            elif self.comb_type == 'Linear':
                comb_str = 'nn.Linear(self.dim * s_arity, self.dim).to(self.device)'
            else:
                raise
            exec('self.symbol.update({str(s) :' + comb_str + '})')
            


        for s_type in self.all_types:
            s_nullary = []            

            for s in sorted(self.symbols[s_type]):
                if s not in self.arity:
                    s_nullary.append(s)
                else:
                    if (s in self.uncommon_trans) and (s != self.uncommon_trans[s]):
                        #self.symbol.update({str(s):self.uncommon_symbols_repr[s_type][s]})
                        #print(self.uncommon_symbols_repr[s_type][s])                        
                        pass
                    else:
                        new_comb(s)

            if s_nullary:
                #self.s_emb = torch.nn.Embedding(len(s_nullary), self.dim).to(self.device)
                #s_emb_pos = self.s_emb(torch.LongTensor([range(len(s_nullary))]).to(self.device))

                #for i, s in enumerate(s_nullary):
                for s in s_nullary:
                    if (s in self.uncommon_trans) and (s != self.uncommon_trans[s]):
                        #self.symbol[str(s)] = self.uncommon_symbols_repr[s_type][s]
                        #print(self.uncommon_symbols_repr[s_type][s])
                        pass
                    else:
                        # SAVE THEM !!!
                        #self.symbol[s] = torch.rand(self.dim, device=self.device, requires_grad=True)#.to(self.device)
                        #self.symbol[s] = self.s_emb(torch.LongTensor([s]))[0]
                        pass
        #if self.use_conjecture:
        #    final_mult = 2
        #else:
        #    final_mult = 1

        if self.final_type == 'LinearReLU':
            self.final = nn.Sequential(nn.Linear(self.conjecture_dim + self.dim, max(2, self.dim // 2)), nn.ReLU(), nn.Linear(max(2, self.dim // 2), 2)).to(self.device)
        elif self.final_type == '3LinearReLU':
            self.final = nn.Sequential(nn.Linear(self.conjecture_dim + self.dim, self.conjecture_dim + self.dim), nn.ReLU(), nn.Linear(self.conjecture_dim + self.dim, max(2, self.dim // 2)), nn.ReLU(), nn.Linear(max(2, self.dim // 2), 2)).to(self.device)
        elif self.final_type == 'LinearTanh':
            self.final = nn.Sequential(nn.Linear(self.conjecture_dim + self.dim, max(2, self.dim // 2)), nn.Tanh(), nn.Linear(max(2, self.dim // 2), 2)).to(self.device)
        elif self.final_type == 'Linear':
            self.final = nn.Linear(self.conjecture_dim + self.dim, 2).to(self.device)
        else:
            raise

        self.clause_bidir = False

        # Change to GRU
        self.clause_net = nn.LSTM(self.dim, self.dim, self.clause_depth,
                                   bidirectional=self.clause_bidir).to(self.device)
        
        # SAVE THEM !!!
        # Zeros due to export problems.
        self.clause_h = nn.Parameter(torch.zeros(self.clause_depth * (int(self.clause_bidir) + 1), 1,
                                      self.dim, device=self.device))
        self.clause_c = nn.Parameter(torch.zeros(self.clause_depth * (int(self.clause_bidir) + 1), 1,
                                      self.dim, device=self.device))


        self.conjecture_bidir = False

        # Change to GRU
        self.conjecture_net = nn.LSTM(self.dim, self.conjecture_dim, self.conjecture_depth,
                                   bidirectional=self.conjecture_bidir).to(self.device)
        
        # SAVE THEM !!!
        # Zeros due to export problems.
        self.conjecture_h = nn.Parameter(torch.zeros(self.conjecture_depth * (int(self.conjecture_bidir) + 1), 1,
                                      self.dim, device=self.device))
        self.conjecture_c = nn.Parameter(torch.zeros(self.conjecture_depth * (int(self.conjecture_bidir) + 1), 1,
                                      self.dim, device=self.device))
        
        
        #self.conjecture_hc = (self.conjecture_h, self.conjecture_c)
        


        
    #def symbol_name(self, string, value):
    #    symbol_i = self.str_to_int[string]
    #
    #    if symbol_i in self.arity:
    #        return self.symbol[symbol_i](value)
    #    else:
    #        return self.symbol[symbol_i]


    def uncommon_val(self, s):
        if s in self.uncommon_trans:
            return self.uncommon_trans[s]
        else:
            return s

    def symbol_value(self, s):
        if s in self.arity:
            return self.symbol[str(self.uncommon_val(s))]
        else:
            return self.s_emb.weight[self.uncommon_val(s)]

    def compute(self, gen_term):
        if (not self.cache) or (gen_term not in self.cache_repr):
        #if True:
            if isinstance(gen_term, tuple):
                assert len(gen_term) > 1

                if self.random_order and (gen_term[0] == self.equality_symb):
                    assert len(gen_term) == 3
                    if random.choice([True, False]):
                        pass
                    else:
                        gen_term = (gen_term[0], gen_term[2], gen_term[1])

                params = torch.cat([self.compute(gen_term[i]) for i in range(1, len(gen_term))]).to(self.device)
                   
                comb = self.symbol_value(gen_term[0])
                #print(gen_term, gen_term[0], self.int_to_str[gen_term[0]], comb)
                
                self.cache_repr[gen_term] = comb(params)
                #output = comb(params)
            else:
                self.cache_repr[gen_term] = self.symbol_value(gen_term)
                #output = self.symbol_value(gen_term)
            
        return self.cache_repr[gen_term]
        #return output

    def export_clause(self, clause_vec):

        output, _ = self.clause_net(clause_vec.view(-1, 1, self.dim))

        if self.clause_bidir:
            formated_out = output[-1][0][self.dim:]
        else:
            formated_out = output[-1][0]

        return formated_out


    def compute_conjecture(self, fileid):
        if fileid not in self.cache_conjecture:
            if self.random_order:
                in_clauses = random.sample(self.fileid_to_conj[fileid], len(self.fileid_to_conj[fileid]))
            else:
                in_clauses = self.fileid_to_conj[fileid]                

            clauses = torch.cat([self.compute_clause(c) for c in in_clauses]).to(self.device).view(len(in_clauses),1,-1)

            output, _ = self.conjecture_net(clauses)

            if self.conjecture_bidir:
                self.cache_conjecture[fileid] = output[-1][0][self.dim:]
            else:
                self.cache_conjecture[fileid] = output[-1][0]
        return self.cache_conjecture[fileid]

    
    def compute_clause(self, clause):
        if self.random_order:
            clause = random.sample(clause, len(clause))
        literals = torch.cat([self.compute(l) for l in clause]).to(self.device).view(len(clause),1,-1)
        
        #output, _ = self.clause_net(literals, (self.clause_h, self.clause_c))
        output, _ = self.clause_net(literals)

        if self.clause_bidir:
            return output[-1][0][self.dim:]
        else:
            return output[-1][0]

    def compute_final(self, clause):
        if not self.use_conjecture:
            clause_vec = self.compute_clause(clause)
            clause_res = self.final(clause_vec)
        else:
            clause_in, fileid = clause
            clause_vec = self.compute_clause(clause_in)
            problem_vec = self.compute_conjecture(fileid)
            final_in = torch.cat([clause_vec, problem_vec]).to(self.device)
            clause_res = self.final(final_in)
            
        return clause_res

    def clear_caches(self):
        self.cache_repr = dict()

        if  self.use_conjecture:
            self.cache_conjecture = dict()

    def forward(self, clauses):
        if self.eval_mode:
            #results = [self.compute_final(x) for x in tqdm(clauses)]
            # if self.parallel:
            #     mp.set_start_method('spawn')
            #     with mp.Pool(processes=self.nthreads) as pool:
            #         result = list(pool.map(self.compute_final, clauses))
            # else:
            results = [self.compute_final(x) for x in clauses]
        else:
            # if self.parallel:
            #     mp.set_start_method('spawn')
            #     with mp.Pool(processes=self.nthreads) as pool:
            #         result = list(pool.map(self.compute_final, clauses))
            # else:
            results = [self.compute_final(x) for x in clauses]

            #output = torch.zeros(len(clauses) * 2).to(self.device)
            #for i, x in enumerate(clauses):
            #    output[i*2:(i*2)+2] = self.compute_final(x)


        output = torch.cat(results).to(self.device)
        return output

        

        
        
        
