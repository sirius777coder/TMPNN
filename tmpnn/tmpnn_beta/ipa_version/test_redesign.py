import json, time, os, sys, glob, copy

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import random

## Library code
import struct2seq
import utils
import protein_features
import data
import utils
# Debug plotting
import matplotlib
import glob
import pandas as pd
from argparse import ArgumentParser

from matplotlib import pyplot as plt
plt.switch_backend('agg')



# Get the input parameters
parser = ArgumentParser(description='Structure to sequence modeling')
parser.add_argument('--data_jsonl', type=str,help='Path for the jsonl data')
parser.add_argument('--split_json', type=str, help='Path for the split json file')
parser.add_argument('--checkpoint',type=str,help="model parameters")
parser.add_argument('--output',default="./",type=str,help="output parameters")
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature to sample an amino acid')
parser.add_argument('--batch_size',type=int,default=7000,help="batch size tokens")
parser.add_argument('--cctop',type=bool,default=True,help="batch size tokens")


args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = struct2seq.TMPNN(device=device) #此处的device是为了初始化device,之后就可以正常调用了
model = model.to(device) #模型参数全部给device
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
criterion = torch.nn.NLLLoss(reduction='none')

# Load the test set from a splits file
with open(args.split_json) as f:
    dataset_splits = json.load(f)
test_names = dataset_splits['test']
# Load the dataset
dataset = data.StructureDataset(jsonl_file=jsonl_file, max_length=args.max_length,high_fraction=args.mask) # total dataset of the pdb files

# 统一所有的数据格式
for i in dataset:
    i['length'] = len(i['seq'])
for i in dataset:
    if 'cctop' not in i.keys():
        i['cctop'] = "I"*i['length']
    i['cctop'] = i['cctop'].replace("T","S") #替换减少一类
for i in dataset:
    if "AF" in i['name']:
        i['name'] += "_A"
# Split the dataset
dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
test_set = Subset(dataset, [dataset_indices[name] for name in test_names])
loader_test = data.StructureLoader(test_set, batch_size=args.batch_size)
print('Testing {} domains'.format(len(test_set)))


def _plot_log_probs(log_probs, total_step):
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    reorder = 'DEKRHQNSTPGAVILMCFWY'
    permute_ix = np.array([alphabet.index(c) for c in reorder])
    plt.close()
    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(111)
    P = np.exp(log_probs.cpu().data.numpy())[0].T
    plt.imshow(P[permute_ix])
    plt.clim(0,1)
    plt.colorbar()
    plt.yticks(np.arange(20), [a for a in reorder])
    ax.tick_params(axis=u'both', which=u'both',length=0, labelsize=5)
    plt.tight_layout()
    plt.savefig(base_folder + 'probs{}.pdf'.format(total_step))
    return

def _loss(S, log_probs, mask):
    """ Negative log probabilities """
    loss = criterion(
        log_probs.contiguous().view(-1,log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


def _scores(S, log_probs, mask):
    """ Negative log probabilities """
    loss = criterion(
        log_probs.contiguous().view(-1,log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores

def recovery(x,y):
    return np.mean([i==j for i,j in zip(x,y)])

def _S_to_seq(S, mask):
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0])
    return seq

def _cctop(C,mask):
    alphabet = "IMOSL"
    cctop = ''.join([alphabet[c] for c, m in zip(C.tolist(), mask.tolist()) if m > 0])
    return cctop


# Build paths for experiment
base_folder = args.output
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
for subfolder in ['alignments']:
    if not os.path.exists(base_folder + subfolder):
        os.makedirs(base_folder + subfolder)
logfile = base_folder + '/log.txt'
with open(base_folder + '/hyperparams.json', 'w') as f:
    json.dump(vars(args), f)


BATCH_COPIES = 10
NUM_BATCHES = 1
# temperatures = [1.0, 0.33, 0.1, 0.033, 0.01]
temperatures = [args.temperature] 

# Timing
start_time = time.time()
total_residues = 0

total_step = 0
# Validation epoch
model.eval()
alphabet = "IMOULS"
with torch.no_grad():
    test_sum, test_weights = 0., 0.
    for ix, protein in enumerate(test_set):
        
        batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
        X, S,C,mask, lengths = utils.featurize(batch_clones, device, shuffle_fraction=0.)
        log_probs,logits_cctop = model(X=X, S=S,L=lengths, mask=mask,device=device)
        logits_cctop_top = logits_cctop[0].unsqueeze(0)
        mask_top = mask[0].unsqueeze(0)
        pred_cctop = (model.decode_crf(logits_cctop_top,mask_top))[0]
        scores = _scores(S, log_probs, mask)
        native_score = scores.cpu().data.numpy()[0]
        print(scores)

        # Generate some sequences
        ali_file = base_folder + 'alignments/' + batch_clones[0]['name'] + '.fa'
        
        with open(ali_file, 'w') as f:
            native_seq = _S_to_seq(S[0], mask[0])
            native_cctop = _cctop(C[0],mask[0])
            pred_cctop = "".join([alphabet[idx] for idx in pred_cctop])
            f.write('>Native, score={}\n{}\n'.format(native_score, native_seq))
            f.write(f">Ncctop,{native_cctop}\n")
            f.write(f">Pcctop,{pred_cctop}\n{np.mean([i==j for i,j in zip(native_cctop,pred_cctop)]) :.3f}\n")
            for temp in temperatures:
                for j in range(NUM_BATCHES):
                    S_sample = model.sample(X, lengths, mask, temperature=temp)

                    # Compute scores

                    log_probs,logits_cctop = model(X, S_sample, lengths, mask,device=device)
                    batch_cctop = model.decode_crf(logits_cctop,mask)

                    scores = _scores(S_sample, log_probs, mask)
                    scores = scores.cpu().data.numpy()

                    for b_ix in range(BATCH_COPIES):
                        seq = _S_to_seq(S_sample[b_ix], mask[0])
                        pred_cctop_idx = batch_cctop[b_ix]
                        pred_cctop = "".join([alphabet[idx] for idx in pred_cctop_idx])
                        score = scores[b_ix]
                        f.write(f'>T={temp},sample={b_ix},score={score},recovery={recovery(seq,native_seq) :.3f},acc={np.mean([i==j for i,j in zip(native_cctop,pred_cctop)]) :.3f}\n{seq}\n{pred_cctop}\n')

                    total_residues += torch.sum(mask).cpu().data.numpy()
                    elapsed = time.time() - start_time
                    residues_per_second = float(total_residues) / float(elapsed)
                    print('{} residues / s'.format(residues_per_second))

                frac_recovery = torch.sum(mask * (S.eq(S_sample).float())) / torch.sum(mask)
                frac_recovery = frac_recovery.cpu().data.numpy()
                # print(mask)
                # print(frac_recovery, torch.numel(mask), torch.sum(mask).cpu().data.numpy(), batch_clones[0]['name'])
                print(frac_recovery)

# Plot the results
files = glob.glob(base_folder + 'alignments/*.fa')
df = pd.DataFrame(columns=['name', 'T', 'score', 'similarity'])

def similarity(seq1, seq2):
    matches = sum([c1==c2 for c1, c2 in zip(seq1,seq2)])
    return float(matches) / len(seq1)

for file in files:
    with open(file, 'r') as f:
        # Skip over native
        entries = f.read().split('>')[1:]
        entries = [e.strip().split('\n') for e in entries]

        # Get native information
        native_header = entries[0][0]
        native_score = float(native_header.split(', ')[1].split('=')[1])
        native_seq = entries[0][1]
        # print(entries[0])
        # print(native_score)

        for header, seq in entries[1:]:
            T, sample, score = [float(s.split('=')[1]) for s in header.split(', ')]
            pdb, chain = file.split('/')[-1].split('.')[0:2]

            df = df.append({
                'name': pdb + '.' + chain,
                'T': T, 'score': score,
                'native': native_score,
                'similarity': similarity(native_seq, seq)
                },  ignore_index=True
            )

df['diff'] = -(df['score'] - df['native'])

boxplot = df.boxplot(column='diff', by= 'T')
plt.xlabel('Decoding temperature')
plt.ylabel('log P(sample) - log P(native)')
boxplot.get_figure().gca().set_title('')
boxplot.get_figure().suptitle('')
plt.tight_layout()
plt.savefig(base_folder + 'decoding.pdf')

boxplot = df.boxplot(column='similarity', by= 'T')
plt.xlabel('Decoding temperature')
plt.ylabel('Native sequence recovery')
boxplot.get_figure().gca().set_title('')
boxplot.get_figure().suptitle('')
plt.tight_layout()
plt.savefig(base_folder + 'recovery.pdf')

# Store the results
df_mean = df.groupby(['name', 'T'], as_index=False).mean()
df_mean.to_csv(base_folder + 'results.csv')

print('Speed total: {} residues / s'.format(residues_per_second))
print('Median', df_mean['similarity'].median())
