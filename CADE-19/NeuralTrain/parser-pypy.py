import argparse
import os
import tqdm
from multiprocessing import Pool
from collections import Counter
from lark import Lark, Transformer
import pickle

import sys
sys.setrecursionlimit(10000)


parser = argparse.ArgumentParser(description = ("Parse the dataset"), formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('dataset',
                    type=str,
                    help='output dataset filename')

parser.add_argument('--dirs',
                    type=str,
                    nargs='*',
                    help='input directories')

parser.add_argument('--problem',
                    type=str,
                    default='train_cnf/bushy_mzr02_10s/cnf/',
                    help='where the source cnfs of the problem are')

parser.add_argument('--prune_filenames',
                    type=str,
                    default='',
                    help='use this filename to pruning')


parser.add_argument('--cpu',
                    type=int,
                    default=0,
                    help='how many cpus use, default max // 2')

parser.add_argument('--problem_only_conjectures',
                    dest='problem_only_conjectures',
                    default=False,
                    action='store_true',
                    help='problem only conjectures - unsafe!!!')

parser.add_argument('--no-problem_only_conjectures',
                    dest='problem_only_conjectures',
                    default=False,
                    action='store_false',
                    help='no problem only conjectures')


parser.add_argument('--prune_problem_only_conjectures',
                    dest='prune_problem_only_conjectures',
                    default=True,
                    action='store_true',
                    help='prune problem only conjectures before pickle')

parser.add_argument('--no-prune_problem_only_conjectures',
                    dest='prune_problem_only_conjectures',
                    default=True,
                    action='store_false',
                    help='prune problem only conjectures before pickle')

parser.add_argument('--use_translation',
                    type=str,
                    default='',
                    help='use translation from a given dataset')


args = parser.parse_args()

POS = 'pos/'
NEG = 'neg/'

if args.cpu:
    proc = args.cpu
else:
    proc = os.cpu_count() // 2

if args.prune_filenames:
    with open(args.prune_filenames, 'r') as f:
        prune_filenames = {x[:-1] for x in f}
else:
    prune_filenames = set()
        
    


class Clause(Transformer):
    cnf_line = tuple
    clause = list
    term = lambda self, x: x[0]
    complex_term = tuple
    epred = lambda self, x: x[0]
    literal = lambda self, x: x[0]
    atom = tuple
    
    line = lambda self, x: x[0][:]
    #p_name = lambda self, x: s_predicate(x[0][:])
    #f_name = lambda self, x: s_function(x[0][:])
    #constant = lambda self, x: s_constant(x[0][:])
    
    #epred = lambda self, x: s_epred('epred_0', data)
    #neg_epred = lambda self, x: ('~', s_epred('epred_0'))
    
    eq = lambda self, x: ('=', x[0], x[1])
    noneq = lambda self, x: ('~', ('=', x[0], x[1]))
    neglit = lambda self, x: ('~', x[0])
    
    
    gen_neg_atom = lambda self, x: ('~', x)
    
    #var = lambda self, _: 'X'
    plain = lambda self, _: False
    negated_conjecture = lambda self, _: True
    false = lambda self, _: "$false"
    true = lambda self, _: "$true"


def load_cnf_line(cnf_line):
    tree = cnf_parser.parse(cnf_line)  
    out = Clause().transform(tree)
    return out


cnf_parser = Lark("""
    cnf_line: "cnf(" line_name "," problem_type "," clause ")."
    
    ?line_name: LINE -> line
    
    ?problem_type: "plain" -> plain
        | "negated_conjecture" -> negated_conjecture
    
    clause: literal ("|" literal)*
        | "(" literal ("|" literal)* ")"
    
    
    ?literal: positive_literal
        | negative_literal
        
    ?negative_literal: "~" positive_literal -> neglit
        | term "!=" term -> noneq
    
    ?positive_literal: gen_atom
        | term "=" term -> eq
        | false
        | true
                
    false: "$false" -> false
    true: "$true" -> true
    
    ?gen_atom : prop_atom
        | atom
    
    ?prop_atom: SKOLEM_PREDICATE
        | PREDICATE
    
    atom: p_name "(" term ("," term)* ")"
    
    ?p_name: SKOLEM_PREDICATE
        | PREDICATE
      
    term : VARIABLE
        | constant
        | complex_term
    
    complex_term: f_name "(" term ("," term)* ")"
    
    VARIABLE: /X/ INT
        
    ?f_name: esk_f_name
        | norm_f_name
    
    ?esk_f_name: SKOLEM_FUNCTION
    
    ?norm_f_name: FUNCTION
    
    ?constant: norm_constant
        | esk_constant
    
    ?esk_constant: SKOLEM_CONSTANT
    
    ?norm_constant: CONSTANT
        | INT
        
    LINE: LCASE_LETTER (LCASE_LETTER | INT | "_")*
    PREDICATE: /(?!epred)/ LCASE_LETTER (LCASE_LETTER | INT | "_")*
    SKOLEM_PREDICATE: /epred/ INT /_/ INT
    FUNCTION: /(?!esk)/ LCASE_LETTER (LCASE_LETTER | INT | "_")*
    SKOLEM_FUNCTION: /esk/ INT /_/ INT
    CONSTANT: /(?!esk)/ LCASE_LETTER (LCASE_LETTER | INT | "_")*
    SKOLEM_CONSTANT: /esk/ INT /_/ /0/
    

    
    %import common.ESCAPED_STRING -> STRING
    %import common.LCASE_LETTER -> LCASE_LETTER
    %import common.INT -> INT
    %import common.CNAME -> CNAME
    %import common.WS
    %ignore WS
    
    """, start='cnf_line')

SPECIAL_TYPES = {'PREDICATE',
                 'SKOLEM_PREDICATE',
                 'FUNCTION',
                 'SKOLEM_FUNCTION',
                 'CONSTANT',
                 'SKOLEM_CONSTANT',
                 'INT',
                 'VARIABLE',
                }

ALL_TYPES = SPECIAL_TYPES | {'other'}

def symbols_in_literal(x, data):
    if isinstance(x, tuple):
        y = x[0]
        if hasattr(y, 'value'):
            y = y.value
        if y not in data['arity']:
            data['arity'][y] = len(x) - 1
        else:
            assert len(x) - 1 == data['arity'][y]
        for y in x:
            symbols_in_literal(y, data)
    elif hasattr(x, 'type') and (x.type in SPECIAL_TYPES):
        data[x.type].update([x.value])
    elif isinstance(x, str):
        data['other'].update([x])       
    else:
        print('Problem with:', x)
        raise

    
def scan_symbols(list_clauses):
    data = dict()
    
    for x in ALL_TYPES:
        data[x] = Counter()
        
    data['arity'] = dict()
    
    for clause in list_clauses:
        for literal in clause[2]:
            symbols_in_literal(literal, data)                   
    
    return data


def load_filename(filename):
    #print(filename)
    with open(filename) as f:
        if only_conj:
            out = [load_cnf_line(x[:-1].split('#')[0]) for x in f if x.startswith('cnf(') and ('negated_conjecture' in x)]
        else:
            out = [load_cnf_line(x[:-1].split('#')[0]) for x in f if x.startswith('cnf(')]
    
    symbols = scan_symbols(out)
    
    problem = filename.split('/')[-1].split('.')[0]
    
    return problem, (out, symbols)


def load_dirs(dirnames):
    filenames = []
    for dirname in dirnames:
        filenames += [dirname + x for x in os.listdir(dirname) if os.path.getsize(dirname + x) > 0]

    if prune_filenames:
        filenames = [x for x in filenames if x.split('/')[-1].split('.')[0] in prune_filenames]
        
    out_dict = dict()
        
    with Pool(processes=proc) as pool:
        if proc == 1:
            iterate_over = map(load_filename, filenames)
        else:
            iterate_over = pool.imap_unordered(load_filename, filenames, chunksize=1)
        for x, y in tqdm.tqdm(iterate_over, total=len(filenames)):
            if x not in out_dict:
                out_dict[x] = y
            else:
                problem, rest = out_dict[x]

                problem += y[0]

                for k in rest:
                    rest[k].update(y[1][k])

                out_dict[x] = (problem, rest)
        
    # for test purposes
    #out_dict = {x: y for x, y in tqdm.tqdm_notebook(map(load_filename, filenames[:10]), total=len(filenames))}
    return out_dict

only_conj = False
p_neg = load_dirs([x + NEG for x in args.dirs])
p_pos = load_dirs([x + POS for x in args.dirs])
only_conj = args.problem_only_conjectures
p_problem = load_dirs([args.problem])


def counters_arity(list_parsed):
    ca_counters = dict()
    ca_arity = dict()

    for type_name in ALL_TYPES:
        ca_counters[type_name] = Counter()
    
    for parsed in list_parsed:
        for filename, parsed_data in parsed.items():
            for d_type, counter in parsed_data[1].items():
                if d_type == 'arity':
                    for s, s_arity in counter.items():
                        if s in ca_arity:
                            assert ca_arity[s] == s_arity
                        else:
                            ca_arity[s] = s_arity
                else:
                    ca_counters[d_type].update(counter)
                    
    return ca_counters, ca_arity


all_counters, all_arity = counters_arity([p_problem, p_pos, p_neg])


for x in all_counters['CONSTANT'] | all_counters['FUNCTION'] | all_counters['PREDICATE']:
    assert (not x.startswith('esk')) and (not x.startswith('epred'))


arities = {x for _, x in all_arity.items()}
print('Arities:', arities)

all_symbols = Counter()
for _, y in all_counters.items():
    all_symbols.update(y)

print('Len all symbols:', len(all_symbols))
print('Len all arity:', len(all_arity))

def simplify(x):
    if x.startswith('esk'):
        y = x.split("_")
        assert len(y) == 2
        return 'esk_' + y[1]
    elif x.startswith('epred'):
        y = x.split("_")
        assert len(y) == 2
        return 'epred_' + y[1]
    elif x.startswith('X'):
        return 'X'
    else:
        return x

# We use simplify on our things!!!
if args.use_translation:
    translation = pickle.load(open(args.use_translation, 'rb'))['str_to_int']
else:
    translation_midstep = {y: x+1 for x, y in enumerate({simplify(s) for s in all_symbols.keys()})}
    translation = {s: translation_midstep[simplify(s)] for s in all_symbols.keys()}

    

assert translation['esk1_0'] == translation['esk2_0']

print('SYMBOLS Before translation:', len(translation), ' After translation:', len({translation[x] for x in translation}))


def translate_literal(x):
    if isinstance(x, tuple):
        return tuple([translate_literal(y) for y in x])
    else:
        return translation[x]

def translate_clause(x):
    assert len(x) == 3
    assert isinstance(x[2], list)
    
    new_x2 = [translate_literal(l) for l in x[2]]
    
    return (x[0], x[1], new_x2)    
    
def translate_file(z):
    name, (clauses, symbols) = z
    new_clauses = [translate_clause(z) for z in clauses]
    
    new_symbols = dict()
    
    for x in ALL_TYPES:
        new_symbols[x] = Counter()
    
    for x, y in symbols.items():
        if x in ALL_TYPES:
            for a, b in y.items():
                new_symbols[x].update(Counter({translation[a]: b}))
        
    new_symbols['arity'] = {translation[x]: y for x, y in symbols['arity'].items()}
    
    return name, (new_clauses, new_symbols)

def translate_problem(p):
    #with Pool(processes=os.cpu_count()) as pool:
    #    out_dict = {x: y for x, y in tqdm.tqdm_notebook(pool.imap_unordered(translate_file, p.items(), chunksize=1), total=len(p))}
    out_dict = {x: y for x, y in tqdm.tqdm(map(translate_file, p.items()), total=len(p))}
    return out_dict


np_problem = translate_problem(p_problem)
np_pos = translate_problem(p_pos)
np_neg = translate_problem(p_neg)


str_to_int = translation

int_to_str = dict()

for y, x in translation.items():
    if x in int_to_str:
        int_to_str[x].append(y)
    else:
        int_to_str[x] = [y]


filenames = set(np_pos.keys()) # & set(np_problem.keys())
print('The number of filenames used:', len(filenames))


test_new_all_counters = dict()
for x, y in all_counters.items():
    test_new_all_counters[x] = Counter()               
    for a, b in y.items():
        test_new_all_counters[x].update({translation[a]: b})


new_all_counters, new_all_arity = counters_arity([np_problem, np_pos, np_neg])


assert new_all_arity == {translation[x]: y for x, y in all_arity.items()}
assert new_all_counters == test_new_all_counters


### PRUNING

s_neg = str_to_int['~']
s_eq = str_to_int['=']

def uniform_literal(literal):
    if isinstance(literal, tuple):
        if literal[0] == s_eq:
            _, x, y = literal
            return frozenset([s_eq, x, y])
        elif literal[0] == s_neg and isinstance(literal[0], tuple):
            if (literal[1][0] == s_eq):
                _, x, y = literal[1]
                return (s_neg, frozenset([s_eq, x, y]))
        else:
            return literal
    else:
        return literal

def uniform_clause(clause):
    return frozenset([uniform_literal(l) for l in clause])

def uniform_conjecture(problem):
    return frozenset([uniform_clause(clause[2]) for clause in problem if clause[1]])
    

conjectures = dict()

for filename, (problem, _)  in np_problem.items():
    conjectures[filename] = uniform_conjecture(problem)

positives = set()

for filename, (problem, _) in np_pos.items():
    for clause in problem:
        positives.add((conjectures[filename], uniform_clause(clause[2])))

for filename, f_tuple in np_neg.items():
    problem, rest = f_tuple
    len_before = len(problem)

    problem = [clause for clause in problem if (conjectures[filename], uniform_clause(clause[2])) not in positives]

    len_after = len(problem)

    np_neg[filename] = (problem, rest)

    print(filename, len_before, len_after, len_before - len_after)

###

if args.prune_problem_only_conjectures:
    np_problem = {x: ([c for c in y[0] if c[1]], y[1]) for x, y in np_problem.items()}

###

dataset = dict()
dataset['int_to_str'] = int_to_str
dataset['str_to_int'] = str_to_int
dataset['positive'] = np_pos
dataset['negative'] = np_neg
dataset['problem'] = np_problem
dataset['symbols'] = new_all_counters
dataset['arity'] = new_all_arity
dataset['filenames'] = filenames
dataset['all_types'] = ALL_TYPES
dataset['dir'] = args.dirs
dataset['problem_source'] = args.problem

dataset_filename = args.dataset

print('Pickling data...')

with open(dataset_filename, 'wb') as f:            
        pickle.dump(dataset, f)







