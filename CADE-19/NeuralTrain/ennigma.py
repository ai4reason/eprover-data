import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import rnnennigma
import pickle
import numpy as np
import random
import math
import pytorch_log
import uuid
import time
import os
from collections import Counter
from tqdm import tqdm

parser = argparse.ArgumentParser(description = ("Run experiments with Ennigma"), formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data',
                    type=str,
                    help='data filename')

default_log_uuid = uuid.uuid4().hex
default_log_dir = 'logs/' + time.strftime("%Y-%m-%d-%H-%M-%S-") + default_log_uuid
default_log_filename = default_log_dir + '/' + default_log_uuid + '.log'
parser.add_argument('--log',
                    type=str,
                    default=default_log_filename,
                    help='log directory')

parser.add_argument('--log_prefix',
                    type=str,
                    default='',
                    help='where logs are')

parser.add_argument('--log_filename',
                    type=str,
                    default='',
                    help='the name of log file')


parser.add_argument('--dim',
                    type=int,
                    default=8,
                    help='the dimension of internal vectors')


parser.add_argument('--batch_print',
                    type=int,
                    default=100,
                    help='print every n batches')

parser.add_argument('--clause_depth',
                    type=int,
                    default=1,
                    help='the depth of clauses --- recurrent')

parser.add_argument('--pred_depth',
                    type=int,
                    default=1,
                    help='the depth of predicates')

parser.add_argument('--fnc_depth',
                    type=int,
                    default=1,
                    help='the depth of functions')

parser.add_argument('--positive',
                    type=float,
                    default=1.0,
                    help='the importance of positive examples is multiplied by that')


parser.add_argument('--negative_as_positive',
                    dest='negative_as_positive',
                    default=False,
                    action='store_true',
                    help='the number of negative examples in one epoch is equal to the number of positive examples')

parser.add_argument('--no-negative_as_positive',
                    dest='negative_as_positive',
                    default=False,
                    action='store_false',
                    help='the number of negative examples in one epoch is not equal to the number of positive examples')



parser.add_argument('--dev',
                    type=str,
                    default='cpu',
                    help='the name of device, e.g. cpu or cuda:1')

parser.add_argument('--lr',
                    type=float,
                    default=1e-3,
                    help='learning rate')

parser.add_argument('--batch',
                    type=int,
                    default=16,
                    help='batch size')

parser.add_argument('--batch_problem',
                    dest='batch_problem',
                    default=False,
                    action='store_true',
                    help='a batch is a problem')

parser.add_argument('--no-batch_problem',
                    dest='batch_problem',
                    default=False,
                    action='store_false',
                    help='a batch is not a problem')


parser.add_argument('--epoch_log_model',
                    type=int,
                    default=1,
                    help='log model every n epoch')


parser.add_argument('--epoch',
                    type=int,
                    default=100,
                    help='number of epochs')

parser.add_argument('--urule_depth',
                    type=int,
                    default=2,
                    help='unary rule depth')


parser.add_argument('--brule_depth',
                    type=int,
                    default=2,
                    help='binary rule depth')

parser.add_argument('--comb_type',
                    type=str,
                    default='LinearReLU',
                    help='what is the type of model of the combining things')


parser.add_argument('--final_type',
                    type=str,
                    default='LinearReLU',
                    help='what is the type of model of the final layer')


parser.add_argument('--min_occurr',
                    type=int,
                    default=10,
                    help='the minimal number of occurrences of a symbol on the training set')

parser.add_argument('--conjecture',
                    dest='conjecture',
                    default=False,
                    action='store_true',
                    help='use conjecture')

parser.add_argument('--no-conjecture',
                    dest='conjecture',
                    default=False,
                    action='store_false',
                    help='use conjecture')

parser.add_argument('--conjecture_depth',
                    type=int,
                    default=1,
                    help='the depth of conjectures --- recurrent')

parser.add_argument('--conjecture_dim',
                    type=int,
                    default=0,
                    help='the dimension of conjectures --- recurrent')

parser.add_argument('--layer_norm',
                    dest='layer_norm',
                    default=False,
                    action='store_true',
                    help='layer norm')

parser.add_argument('--no-layer_norm',
                    dest='layer_norm',
                    default=False,
                    action='store_false',
                    help='no layer norm')

parser.add_argument('--split_problem',
                    dest='split_problem',
                    default=True,
                    action='store_true',
                    help='split by problems')

parser.add_argument('--no-split_problem',
                    dest='split_problem',
                    default=True,
                    action='store_false',
                    help='split by fraction over all problems')

parser.add_argument('--cache',
                    dest='cache',
                    default=True,
                    action='store_true',
                    help='cache')

parser.add_argument('--no-cache',
                    dest='cache',
                    default=True,
                    action='store_false',
                    help='cache')


parser.add_argument('--full_model',
                    dest='full_model',
                    default=False,
                    action='store_true',
                    help='full model')

parser.add_argument('--no-full_model',
                    dest='full_model',
                    default=False,
                    action='store_false',
                    help='no full model')


parser.add_argument('--random_order',
                    dest='random_order',
                    default=True,
                    action='store_true',
                    help='random order of clauses, literals, and =')

parser.add_argument('--no-random_order',
                    dest='random_order',
                    default=True,
                    action='store_false',
                    help='fixed order of clauses, literals, and =')



parser.add_argument('--valid_frac',
                    type=float,
                    default=0.0,
                    help='valid fraction')


parser.add_argument('--train_problems',
                    type=str,
                    nargs='*',
                    help='use only selected problems for training')

parser.add_argument('--valid_problems',
                    type=str,
                    nargs='*',
                    help='use only selected problems for validation')

parser.add_argument('--eval_model',
                    type=str,
                    default='',
                    help='evaluate using the given model without final extension')


parser.add_argument('--nthreads',
                    type=int,
                    default=1,
                    help='the number of threads used for evaluating parse trees')


args = parser.parse_args()


if args.log == default_log_filename:

    if args.log_prefix:
        args.log = args.log_prefix + args.log
        os.mkdir(args.log_prefix + default_log_dir)
    else:
        os.mkdir(default_log_dir)

if args.log_filename:
    args.log = args.log + args.log_filename
        
logger = pytorch_log.PyTorchLog(__file__, args.log)

for a, v in args._get_kwargs():
    logger.kwarg(a, v)


logger.info('Loading data... %s', args.data)
data = pickle.load(open(args.data, 'rb'))
logger.info('Loaded.')

device = torch.device(args.dev if torch.cuda.is_available() else "cpu")

model_params = dict()
model_params['device'] = device
model_params['dimension'] = args.dim
model_params['int_to_str'] = data['int_to_str']
model_params['str_to_int'] = data['str_to_int']
model_params['arity'] = data['arity']
model_params['symbols'] = data['symbols']

model_params['random_order'] = args.random_order

model_params['comb_type'] = args.comb_type
model_params['final_type'] = args.final_type

model_params['conjecture'] = args.conjecture

model_params['all_types'] = data['all_types']

model_params['clause_depth'] = args.clause_depth
model_params['conjecture_depth'] = args.conjecture_depth
model_params['conjecture_dim'] = args.conjecture_dim
model_params['pred_depth'] = args.pred_depth
model_params['fnc_depth'] = args.fnc_depth
model_params['cache'] = args.cache

model_params['nthreads'] = args.nthreads

all_files = data['filenames']

all_files_to_id = {x: y for y, x in enumerate(all_files)}
all_files_from_id = {y: x for x, y in all_files_to_id.items()}

model_params['all_files_from_id'] = all_files_from_id

if args.eval_model:
    args.valid_frac = 1.0
    

if model_params['conjecture']:
    fileid_to_conj = dict()
    if model_params['conjecture']:
        fileid_to_conj = dict()
        for i, s in all_files_from_id.items():
            fileid_to_conj[i] = []

            for x in data['problem'][s][0]:
                if x[1]:
                    fileid_to_conj[i].append(x[2])

    model_params['fileid_to_conj'] = fileid_to_conj
          

def get_pos_neg(files):
    pos = [(x, True, all_files_to_id[f]) for f in files for x in data['positive'][f][0]]
    n_files = [f for f in files if f in data['negative']]
    neg = [(x, False, all_files_to_id[f]) for f in n_files for x in data['negative'][f][0]]
    return pos, neg

if args.valid_frac:
    valid_files = set(random.sample(all_files, int(args.valid_frac * len(all_files))))
    train_files = all_files - valid_files
    valid_pos, valid_neg = get_pos_neg(valid_files)
else:
    train_files = all_files

if args.train_problems:
    train_files = args.train_problems
    logger.info('Training problems %s', train_files)

if args.valid_problems:
    valid_files = args.valid_problems
    logger.info('Validation problems %s', valid_files)
    valid_pos, valid_neg = get_pos_neg(valid_files)
    
train_pos, train_neg = get_pos_neg(train_files)
    

train_symbols = dict()


# Note that we count only symbols that occur in all three sets!!!

def get_used_symbols_counts(filenames):
    count_symbols = dict()
    
    for s_type in model_params['all_types']:
        count_symbols[s_type] = Counter()
        
        for f in filenames:
            count_symbols[s_type].update(data['positive'][f][1][s_type])
            if f in data['negative']:
                count_symbols[s_type].update(data['negative'][f][1][s_type])
            # Exclude problem???
            count_symbols[s_type].update(data['problem'][f][1][s_type])

    return count_symbols

train_symbols = get_used_symbols_counts(train_files)
all_symbols = get_used_symbols_counts(all_files)

model_params['train_symbols'] = train_symbols

uncommon_symbols = dict()


for s_type in data['symbols']:
    uncommon_symbols[s_type] = {s: (s_count, all_symbols[s_type][s], train_symbols[s_type][s]) for s, s_count in model_params['symbols'][s_type].items() if train_symbols[s_type][s] < args.min_occurr}

    logger.info('Uncommon symbols on the training set %s %s %s', s_type, len(uncommon_symbols[s_type]), [(model_params['int_to_str'][s], s_counts) for s, s_counts in uncommon_symbols[s_type].items()])
    
# model_params['uncommon_symbols'] = uncommon_symbols

uncommon_symbols_repr = dict()

for s_type, symbols in uncommon_symbols.items():
    uncommon_symbols_repr[s_type] = dict()
    type_arity_repr = dict()

    for s in sorted(symbols):
        if s in model_params['arity']:
            s_arity = model_params['arity'][s]
        else:
            s_arity = 0

        if s_arity not in type_arity_repr:
            type_arity_repr[s_arity] = s

        uncommon_symbols_repr[s_type][s] = type_arity_repr[s_arity]

model_params['uncommon_symbols_repr'] = uncommon_symbols_repr
model_params['uncommon_trans'] = {x: y for _, z in uncommon_symbols_repr.items() for x, y in z.items()}

logger.info('Uncommon symbols repr: %s', uncommon_symbols_repr)

used_symbols = dict()
for s_type in model_params['all_types']:
    used_symbols[s_type] = set()

    for s in data['symbols'][s_type]:
        if s not in uncommon_symbols_repr[s_type]:
            used_symbols[s_type].add(s)
        else:
            used_symbols[s_type].add(uncommon_symbols_repr[s_type][s])

model_params['used_symbols'] = used_symbols

symbol_to_type = {s: t for t, x in data['symbols'].items() for s in x}

model_params['symbol_to_type'] = symbol_to_type





n_pos = len(train_pos)

if args.negative_as_positive:
    n_neg = n_pos
else:
    n_neg = len(train_neg)

if n_pos:
    negpos_ratio = n_neg / n_pos
else:
    negpos_ratio = 1.0

negpos_ratio_examples = math.ceil(negpos_ratio)    

if args.negative_as_positive:
    loss_weight = torch.tensor([1, args.positive]).to(device)
else:
    loss_weight = torch.tensor([1, negpos_ratio * args.positive]).to(device)


loss_function = nn.CrossEntropyLoss(weight=loss_weight)

if not args.eval_model:
    model = rnnennigma.RNNennigma(model_params)#.to(device)
else:
    model_params = pickle.load(open(args.eval_model + '.model_args', 'rb'))[1]
    model_params['device'] = device
    model_params['eval_mode'] = True
    model = rnnennigma.RNNennigma(model_params).to(device)
    model.load_state_dict(torch.load(args.eval_model + '.model_parameters', map_location=device))
    model.eval()



torch.cuda.empty_cache()

optimizer = optim.Adam(model.parameters(), lr=args.lr)

logger.info('Positive examples: %s', n_pos)
logger.info('Negative examples: %s', n_neg)
logger.info('Neg/pose ratio: %s %s', negpos_ratio, negpos_ratio_examples)
logger.info('#Batches/Epoch: %s', (n_pos + n_neg) / args.batch)




def log_model(model_text):
    model.clear_caches()

    model_filename = args.log + '.' + model_text
    torch.save(model.state_dict(), model_filename + '.model_parameters')
    pickle.dump((device, model_params), open(model_filename + '.model_args', 'wb'))

    if args.full_model:
        torch.save(model, model_filename + '.model')

    #false_input = dict()
    #if args.false_input_type == 'Embedding':
    #        false_input['false_input_emb'] = model.false_input_emb
    #        false_input['false_input_index'] = model.false_input_index
    #        false_input['false_input_eval'] = model.false_input_eval
    #elif args.false_input_type == 'Rand':
    #        false_input['false_input_rand'] = model.false_input_rand
    #        false_input['false_input_eval'] = model.false_input_eval
    #else:
    #    raise
    #pickle.dump(false_input, open(model_filename + '.false_input', 'wb'))

    logger.info('Model %s logged.', model_text)


train_all_batch_problem = dict()

for x in train_pos + train_neg:
    if x[2] not in train_all_batch_problem:
        train_all_batch_problem[x[2]] = [x]
    else:
        train_all_batch_problem[x[2]].append(x)

train_files_ids = [all_files_to_id[f] for f in train_files]

posneg_ratio = n_pos / len(train_neg)

def give_batches():
    if args.batch_problem:
        train_files_random = random.sample(train_files_ids, len(train_files_ids))

        if args.negative_as_positive:
            examples = [x for f in train_files_random for x in random.sample(train_all_batch_problem[f], len(train_all_batch_problem[f])) if (x[1] or random.random() < posneg_ratio)]           
        else:
            examples = [x for f in train_files_random for x in random.sample(train_all_batch_problem[f], len(train_all_batch_problem[f]))]

    else:
        pos_examples = random.sample(train_pos, n_pos)

        if args.negative_as_positive:
            neg_examples = random.sample(train_neg, n_pos)
        else:
            neg_examples = random.sample(train_neg, n_neg)

        n_examples = len(pos_examples) + len(neg_examples)

        examples = random.sample(pos_examples + neg_examples, n_examples)

    ind_range = range(0, len(examples), args.batch)

    for i in tqdm(random.sample(ind_range, len(ind_range))):
        yield examples[i:i+args.batch]


stat_losses = []
stat_predictions = []
stat_labels = []

epoch_stat_losses = []
epoch_stat_predictions = []
epoch_stat_labels = []
    ### Clean data for memory reasons
    ###del data
    ### !!!
    
#with torch.cuda.profiler.profile():
if True:    
    #with torch.autograd.profiler.emit_nvtx():
    if True:
        try:
            for epoch in range(args.epoch):

                for i, batch in enumerate(give_batches()):

                    labels = torch.tensor([int(x[1]) for x in batch]).to(device)

                    model.zero_grad()
                    # Some tensors are not part of the model!!!
                    optimizer.zero_grad()

                    if args.cache:
                        model.clear_caches()            

                    if model_params['conjecture']:
                        predictions = model([(x[0][2], x[2]) for x in batch])
                    else:
                        predictions = model([x[0][2] for x in batch])


                    loss = loss_function(predictions.view(-1,2), labels)


                    stat_losses.append(loss.data.tolist())
                    stat_predictions += (predictions[::2] < predictions[1::2]).data.tolist()
                    stat_labels += labels.data.tolist()


                    if i % args.batch_print == 0:
                        mean_loss = np.mean(stat_losses)

                        local_n_pos = sum(stat_labels)
                        local_n_neg = len(stat_labels) - local_n_pos
                        corr_pos = sum([p == 1 for p, l in  zip(stat_predictions, stat_labels) if l == 1])
                        corr_neg = sum([p == 0 for p, l in  zip(stat_predictions, stat_labels) if l == 0])

                        if local_n_pos:
                            suc_pos = corr_pos / local_n_pos
                        else:
                            suc_pos = 9.9
                        if local_n_neg:                
                            suc_neg = corr_neg / local_n_neg
                        else:
                            suc_neg = 9.9

                        logger.info('Epoch: %s Batch: %s Loss: %s Pos: %s %s Neg: %s %s', epoch, i, "{0:.8f}".format(mean_loss), local_n_pos, "{0:.4f}".format(suc_pos), local_n_neg, "{0:.4f}".format(suc_neg))

                        logger.batch_num(i)
                        logger.batch_loss(str(mean_loss))

                        epoch_stat_losses += stat_losses
                        epoch_stat_predictions += stat_predictions
                        epoch_stat_labels += stat_labels

                        stat_losses = []
                        stat_predictions = []
                        stat_labels = []

                        if False:
                            for test_symbol in ['X1', 'esk1_0']:
                                s_int = model.uncommon_val(model_params['str_to_int'][test_symbol])
                                s_test = model.symbol_value(s_int)
                                print(s_int, s_test)

                            #print(model.clause_hc)

                    loss.backward()
                    #loss.backward(retain_graph=True)
                    optimizer.step()


                if epoch % args.epoch_log_model == 0:
                    log_model(str(epoch))


                with torch.no_grad():

                    epoch_stat_losses += stat_losses
                    epoch_stat_predictions += stat_predictions
                    epoch_stat_labels += stat_labels

                    epoch_mean_loss = np.mean(epoch_stat_losses)

                    local_n_pos = sum(epoch_stat_labels)
                    local_n_neg = len(epoch_stat_labels) - local_n_pos
                    corr_pos = sum([p == 1 for p, l in  zip(epoch_stat_predictions, epoch_stat_labels) if l == 1])
                    corr_neg = sum([p == 0 for p, l in  zip(epoch_stat_predictions, epoch_stat_labels) if l == 0])

                    if local_n_pos:
                        suc_pos = corr_pos / local_n_pos
                    else:
                        suc_pos = 9.9
                    if local_n_neg:                
                        suc_neg = corr_neg / local_n_neg
                    else:
                        suc_neg = 9.9

                    logger.info('TRAIN-EPOCH Epoch: %s Loss: %s Pos: %s %s Neg: %s %s', epoch, "{0:.8f}".format(epoch_mean_loss), local_n_pos, "{0:.4f}".format(suc_pos), local_n_neg, "{0:.4f}".format(suc_neg))

                    epoch_stat_losses = []
                    epoch_stat_predictions = []
                    epoch_stat_labels = []


                    if args.valid_frac:

                        if args.cache:
                            model.clear_caches()            


                        val_data = valid_pos + valid_neg

                        val_labels = torch.tensor([int(x[1]) for x in val_data]).to(device)


                        if model_params['conjecture']:
                            val_predictions = model([(x[0][2], x[2]) for x in val_data])
                        else:
                            val_predictions = model([x[0][2] for x in val_data])

                        val_loss = loss_function(val_predictions.view(-1,2), val_labels)


                        vstat_losses = [val_loss.data.tolist()]
                        vstat_predictions = (val_predictions[::2] < val_predictions[1::2]).data.tolist()
                        vstat_labels = val_labels.data.tolist()



                        val_mean_loss = np.mean(vstat_losses)

                        local_n_pos = sum(vstat_labels)
                        local_n_neg = len(vstat_labels) - local_n_pos
                        corr_pos = sum([p == 1 for p, l in  zip(vstat_predictions, vstat_labels) if l == 1])
                        corr_neg = sum([p == 0 for p, l in  zip(vstat_predictions, vstat_labels) if l == 0])

                        if local_n_pos:
                            suc_pos = corr_pos / local_n_pos
                        else:
                            suc_pos = 9.9
                        if local_n_neg:                
                            suc_neg = corr_neg / local_n_neg
                        else:
                            suc_neg = 9.9

                        if args.eval_model:
                            logger.info('EVAL Epoch: %s Loss: %s Pos: %s %s Neg: %s %s', epoch, "{0:.8f}".format(val_mean_loss), local_n_pos, "{0:.4f}".format(suc_pos), local_n_neg, "{0:.4f}".format(suc_neg))
                        else:
                            logger.info('VALID Epoch: %s Loss: %s Pos: %s %s Neg: %s %s', epoch, "{0:.8f}".format(val_mean_loss), local_n_pos, "{0:.4f}".format(suc_pos), local_n_neg, "{0:.4f}".format(suc_neg))




        except KeyboardInterrupt:
            log_model('ctrlc')

