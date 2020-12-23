from __future__ import division
from __future__ import print_function

import time
import random
import argparse
import math
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score,accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler

from utils_s import load_data, accuracy, F1
from models import GCN,NetD,MLP
from sklearn.preprocessing import MinMaxScaler

# cd /mnt/f/Reikun/pku/lab/DANE/pygcn
# python train_single.py
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--reconstruct_batch', type=int, default=64,
                    help='batch size for l3')
parser.add_argument('--lra',type=float, default=0.2,
                    help='Ratio of labeled nodes in A.')
parser.add_argument('--lrb',type=float, default=0.2,
                    help='Ratio of labeled nodes in A.')
parser.add_argument('--lamb', type=float, default=1.0,
                    help='Coefficient of GAN.')
parser.add_argument('--gamma', type=float, default=10.0,
                    help='Coefficient of cross_entropy.')
parser.add_argument('--theta', type=float, default=10.0,
                    help='Coefficient of loss_center.')
parser.add_argument('--trial',type=int, default=1,
                    help='Trial times.')
parser.add_argument('--type', type=str, default="DANE",
                    help='Type.')

args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda = False
lra = args.lra
lrb = args.lrb
Lambda = args.lamb
Theta = args.theta
Gamma = args.gamma
rbatch_size = args.reconstruct_batch
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
tr = args.trial

one = torch.FloatTensor([1])
mone = one * -1.0
if args.cuda:
    one = one.cuda()
    mone = mone.cuda()
adjA, featuresA, labelsA, idx_trainA, idx_valA, idx_testA, edgesA, edges_weightA, nodes_weightA = load_data(path="../data/transfer/", dataset="usa",preserve_order=1)
adjB, featuresB, labelsB, idx_trainB, idx_valB, idx_testB, edgesB, edges_weightB, nodes_weightB = load_data(path="../data/transfer/", dataset="chn",preserve_order=1)

n_label = max(labelsA)+1
nA = len(labelsA)
nB = len(labelsB)
nodeA = torch.tensor([1.0 for i in range(0,len(labelsA))])
nodeB = torch.tensor([1.0 for i in range(0,len(labelsB))])
if args.type in ['DANE','DANE_noGAN','gsA', 'gsB']:
    ulnodeA = [i for i in range(len(labelsA))]
    ulnodeB = [i for i in range(len(labelsB))]
    lnodeA = None
    lnodeB = None
else:
    lnodeA = torch.multinomial(nodeA, int((lra+0.2)*len(labelsA)), replacement=False)
    ulnodeA = np.setdiff1d(np.array([i for i in range(len(labelsA))]), lnodeA.numpy())
    np.random.shuffle(ulnodeA)
    valnodeA = ulnodeA[0:int(0.2*nA)]
    ulnodeA = ulnodeA[int(0.2*nA)+1:-1]
    print(valnodeA, ulnodeA)
    lnodeB = torch.multinomial(nodeB, int((lrb+0.2)*len(labelsB)), replacement=False)
    ulnodeB = np.setdiff1d(np.array([i for i in range(len(labelsB))]), lnodeB.numpy())
    np.random.shuffle(ulnodeB)
    valnodeB = ulnodeB[0:int(0.2*nB)]
    ulnodeB = ulnodeB[int(0.2*nB)+1:-1]

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic =True
torch.backends.cudnn.enabled = False

# Model and optimizer
mlp = MLP(in_features = args.hidden, nclass = labelsA.max().item() + 1)
Dnet = NetD(nhid=args.hidden)
model = GCN(nfeat=featuresA.shape[1],
            nhid=args.hidden,
            nclass=labelsA.max().item() + 1,
            dropout=args.dropout)
#model.load_state_dict(torch.load('init2.pkl'))
#for item in model.parameters():
#    print(item)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
optimizer_mlp = optim.Adam(mlp.parameters(),
                       lr=0.01, weight_decay=args.weight_decay)
dis_optimizer = optim.SGD(Dnet.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


if args.cuda:
    model.cuda()
    Dnet.cuda()
    featuresA = featuresA.cuda()
    featuresB = featuresB.cuda()
    adjA = adjA.cuda()
    adjB = adjB.cuda()
    labelsA = labelsA.cuda()
    labelsB = labelsB.cuda()

ones = torch.tensor([1.0 for i in range(8*rbatch_size)])

def loss_rec(embeds1, embeds2, neg_sap):
    loss = 0.0
    pos = -torch.mean(F.logsigmoid(torch.sum(embeds1.mul(embeds2),1)))
    
    #print("pos")
    #print(loss)
    neg = 0.0
    for j in range(0,len(neg_sap)):
        tmp= -torch.mean(F.logsigmoid(-torch.sum(embeds1.mul(neg_sap[j]),1)))
        #print('neg')
        #print(tmp)
        neg += tmp
    return pos,neg

def cal_cluster(labelsA, embA, labelsB, embB):
    loss = 0.0
    
    for i in range(n_label):
        idxA = np.where(labelsA == i)
        idxB = np.where(labelsB == i)
        if (idxA[0].size > 0 and idxB[0].size > 0):
            loss += torch.sum((torch.mean(embA[idxA], dim=0) - torch.mean(embB[idxB], dim = 0))**2)
    return loss/n_label
    
def cal_cluster_weight(labelsA, pseu_labelsA, embA_n, embA_u, labelsB, pseu_labelsB, embB_n, embB_u):
    loss = 0.0
    
    for i in range(n_label):
        idxA = np.where(labelsA == i)
        idxB = np.where(labelsB == i)
        centerA = (torch.sum(embA_n[idxA]) + torch.matmul(pseu_labelsA[:,i].unsqueeze(0), embA_u)) / nA
        centerB = (torch.sum(embB_n[idxB]) + torch.matmul(pseu_labelsB[:,i].unsqueeze(0), embB_u)) / nB
        loss += torch.sum((centerA - centerB) ** 2)
    
    return loss/n_label
 

def MLP_train(epoch_num, featuresrc, featuretgt,labelsrc,labeltgt):
    for i in range(epoch_num):
        optimizer_mlp.zero_grad()
        outputsrc = mlp(featuresrc)
        loss_train = F.nll_loss(outputsrc, labelsrc)
        loss_train.backward()
        optimizer_mlp.step()
    
    f1_src = F1(outputsrc, labelsrc)
    acc_src = accuracy(outputsrc, labelsrc)
    outputtgt = mlp(featuretgt)
    f1_tgt = F1(outputtgt, labeltgt)
    acc_tgt = accuracy(outputtgt, labeltgt)
    res = "f1 score: " + str(f1_src) + " acc score: "+ str(acc_src) + " f1 score: "+str(f1_tgt) + " acc score: "+str(acc_tgt) + '\n'
    return res


def train_d(epoch):
    t = time.time()
    model.eval()
    Dnet.train()
    dis_optimizer.zero_grad()
    outputA, embA = model(featuresA, adjA)
    outputB, embB = model(featuresB, adjB)

    idx_trainA = torch.multinomial(nodeA, 8*rbatch_size, replacement=True)
    idx_trainB = torch.multinomial(nodeB, 8*rbatch_size, replacement=True)

    xreal = embA[idx_trainA]
    pred_real = (Dnet(xreal)-ones)**2
    pred_real = pred_real.mean()
    pred_real *= 2.0
    pred_real.backward()

    xfake = embB[idx_trainB]
    pred_fake = Dnet(xfake)**2
    pred_fake = pred_fake.mean()
    pred_fake *= 2.0
    pred_fake.backward()
    D_cost = pred_fake + pred_real
    Wasserstein_D = pred_real - pred_fake
    dis_optimizer.step()

    print('Epoch: {:04d}'.format(epoch+1),
          'cost_dis: {:.4f}'.format(D_cost.item()),
          'loss:{:.4f}'.format(Wasserstein_D),
          'time: {:.4f}s'.format(time.time() - t))



def train_g(epoch, mode = 'GAN', lamb=0.0, theta=0.0, gamma=0.0, flag_display=True):
    t = time.time()
    k = 5

    model.train()
    Dnet.eval()
    sample_edgeA = torch.multinomial(edges_weightA, rbatch_size, replacement=False)
    idx_uA = [edgesA[i][0] for i in sample_edgeA]
    idx_vA = [edgesA[i][1] for i in sample_edgeA]
    sample_edgeB = torch.multinomial(edges_weightB, rbatch_size, replacement=False)
    idx_uB = [edgesB[i][0] for i in sample_edgeB]
    idx_vB = [edgesB[i][1] for i in sample_edgeB]

    #idx_train = torch.multinomial(nodeA, 2*rbatch_size, replacement=True)
    #idx_train = torch.LongTensor(range(140))

    idx_trainA = torch.multinomial(nodeA, 8*rbatch_size, replacement=False)
    idx_trainB = torch.multinomial(nodeB, 8*rbatch_size, replacement=False)

    optimizer.zero_grad()
    outputA, embA = model(featuresA, adjA)
    outputB, embB = model(featuresB, adjB)
    
    xreal = embA[idx_trainA]
    pred_real = Dnet(xreal)**2
    pred_real = pred_real.mean()

    xfake = embB[idx_trainB]
    pred_fake = (Dnet(xfake)-ones)**2
    pred_fake = pred_fake.mean()
    loss_gan = pred_real + pred_fake

    negA = [embA[torch.multinomial(nodes_weightA, rbatch_size, replacement=False)] for i in range(0, k)]
    negB = [embB[torch.multinomial(nodes_weightB, rbatch_size, replacement=False)] for i in range(0, k)]

    loss_reconstructionA_pos, loss_reconstructionA_neg = loss_rec(embA[idx_uA],embA[idx_vA],negA)
    loss_reconstructionB_pos, loss_reconstructionB_neg = loss_rec(embB[idx_uB], embB[idx_vB], negB)
    loss_reconstructionA = loss_reconstructionA_pos + loss_reconstructionA_neg
    loss_reconstructionB = loss_reconstructionB_pos + loss_reconstructionB_neg

    print('lossA =', loss_reconstructionA)
    print('lossB =', loss_reconstructionB)

    if not args.type in ['DANE','DANE_noGAN','gsA', 'gsB']:
        loss_semiA = F.nll_loss(F.log_softmax(outputA, dim=1)[lnodeA], labelsA[lnodeA])
        loss_semiB = F.nll_loss(F.log_softmax(outputB, dim=1)[lnodeB], labelsB[lnodeB])
        loss_cluster = cal_cluster(labelsA[lnodeA], embA[lnodeA], labelsB[lnodeB], embB[lnodeB])
        '''loss_cluster = cal_cluster_weight(labelsA[lnodeA], F.softmax(outputA, dim=1)[ulnodeA], embA[lnodeA], embA[ulnodeA],
                   labelsB[lnodeB], F.softmax(outputB, dim=1)[ulnodeB], embB[lnodeB], embB[ulnodeB])'''

    mode = args.type
    if mode == 'DANE_noGAN':
        loss_train = 10.0*(loss_reconstructionA + loss_reconstructionB)
    elif mode == 'semihgsA':
        loss_train = 10.0*(loss_reconstructionA + loss_semiA)
    elif mode == 'semigsA':
        loss_train = 10.0*(loss_reconstructionA) + theta * (loss_semiA + loss_semiB)
    elif mode == 'semihgsB':
        loss_train = 10.0*(loss_reconstructionB + loss_semiB)
    elif mode == 'semigsB':
        loss_train = 10.0*(loss_reconstructionB) + theta * (loss_semiA + loss_semiB)
    elif mode == 'gsA':
        loss_train = 10.0*loss_reconstructionA
    elif mode == 'gsB':
        loss_train = 10.0*loss_reconstructionB
    elif mode == 'DANE':
        loss_train = 10.0*(loss_reconstructionA + loss_reconstructionB) + lamb * loss_gan
    elif mode == 'cenDANE':
        loss_train = 10.0*(loss_reconstructionA + loss_reconstructionB) + lamb * loss_gan + gamma * loss_cluster
    elif mode == 'semiDANE':
        loss_train = 10.0*(loss_reconstructionA + loss_reconstructionB) + theta * (loss_semiA + loss_semiB) + lamb*loss_gan 
    elif mode == 'semicenDANE':
        loss_train = 10.0*(loss_reconstructionA + loss_reconstructionB) + theta * (loss_semiA + loss_semiB) + lamb*loss_gan + gamma * loss_cluster
    
    print("loss_train =", loss_train)
    loss_train.backward()
    optimizer.step()
    




# Train model
f = open("cite_result/single_{}_lra{}_lrb{}_lamb{}_theta{}_gamma{}_{}.txt".format(args.type,lra,lrb,Lambda,Theta,Gamma,tr),'w')
t_total = time.time()

for epoch in range(args.epochs):
    if not (args.type in ['semigsA', 'semigsB', 'gsA', 'gsB']):
        for i in range(0,5):
            train_d(epoch)
    if epoch % 100 == 0:
        f.write(str(epoch) + ':\n')
        """
        # B->A
        """
        output, feasB = model(featuresB, adjB)
        feasB.detach_()
        feasB = np.array(feasB)
        
        clf = SGDClassifier(loss='log', max_iter=1000, random_state=args.seed)
        clf.fit(feasB, labelsB)
        pred_b = clf.predict(feasB)
        output = np.array(output.detach_())
        output = np.argmax(output, axis = 1)
        
        f1_micro = f1_score(labelsB, pred_b, average = 'micro')
        f1_macro = f1_score(labelsB, pred_b, average = 'macro')
        f.write("B->A: ")
        f.write("f1 score:"+str(f1_macro) + ' ')
        f.write("acc score:"+str(f1_micro) + ' ')
        print("f1 score:" + str(f1_macro))
        print("acc score:" + str(f1_micro))
        
        outputA, feasA = model(featuresA, adjA)
        feasA.detach_()
        feasA = np.array(feasA)
        
        pred_val = clf.predict(feasA[valnodeA])
        f1_micro = f1_score(labelsA[valnodeA], pred_val, average = 'micro')
        f1_macro = f1_score(labelsA[valnodeA], pred_val, average = 'macro')
        f.write("val f1 score:"+str(f1_macro) + ' ')
        f.write("val acc score:"+str(f1_micro) + '\n')
        print("val f1 score:" + str(f1_macro))
        print("val acc score:" + str(f1_micro))
        
        pred_train = clf.predict(feasA[ulnodeA])
        f1_micro = f1_score(labelsA[ulnodeA], pred_train, average = 'micro')
        f1_macro = f1_score(labelsA[ulnodeA], pred_train, average = 'macro')
        f.write("test f1 score:"+str(f1_macro) + ' ')
        f.write("test acc score:"+str(f1_micro) + '\n')
        print("test f1 score:" + str(f1_macro))
        print("test acc score:" + str(f1_micro))
        
        # A->B
        
        clf = SGDClassifier(loss='log', max_iter=1000, random_state=args.seed)
        clf.fit(feasA, labelsA)
        pred_train = clf.predict(feasA)
        f1_micro = f1_score(labelsA, pred_train, average = 'micro')
        f1_macro = f1_score(labelsA, pred_train, average = 'macro')
        f.write("A->B: ")
        f.write("f1 score:"+str(f1_macro) + ' ')
        f.write("acc score:"+str(f1_micro) + ' ')
        print("f1 score:" + str(f1_macro))
        print("acc score:" + str(f1_micro))
        
        pred_val = clf.predict(feasB[valnodeB])
        f1_micro = f1_score(labelsB[valnodeB], pred_val, average = 'micro')
        f1_macro = f1_score(labelsB[valnodeB], pred_val, average = 'macro')
        f.write("val f1 score:"+str(f1_macro) + ' ')
        f.write("val acc score:"+str(f1_micro) + '\n')
        print("val f1 score:" + str(f1_macro))
        print("val acc score:" + str(f1_micro))
        
        pred_b = clf.predict(feasB[ulnodeB])
        f1_micro = f1_score(labelsB[ulnodeB], pred_b, average = 'micro')
        f1_macro = f1_score(labelsB[ulnodeB], pred_b, average = 'macro')
        f.write("test f1 score:"+str( f1_macro) + ' ')
        f.write("test acc score:"+str( f1_micro) + '\n')
        print("test f1 score:" + str(f1_macro))
        print("test acc score:" + str(f1_micro))
        
    #torch.save(model.state_dict(), 'initclear.pkl')
    train_g(epoch,lamb=Lambda,theta=Theta,gamma=Gamma)

# final result
f.write("final:\n")
epoch = 0
output, feasB = model(featuresB, adjB)
feasB.detach_()
feasB = np.array(feasB)
        
clf = SGDClassifier(loss='log', max_iter=1000, random_state=args.seed)
clf.fit(feasB, labelsB)
pred_b = clf.predict(feasB)
f1_micro = f1_score(labelsB, pred_b, average = 'micro')
f1_macro = f1_score(labelsB, pred_b, average = 'macro')
f.write("B->A: ")
f.write("f1 score:"+str(f1_macro) + ' ')
f.write("acc score:"+str(f1_micro) + ' ')
print("f1 score:" + str(f1_macro))
print("acc score:" + str(f1_micro))
        
outputA, feasA = model(featuresA, adjA)
feasA.detach_()
feasA = np.array(feasA)
        
pred_val = clf.predict(feasA[valnodeA])
f1_micro = f1_score(labelsA[valnodeA], pred_val, average = 'micro')
f1_macro = f1_score(labelsA[valnodeA], pred_val, average = 'macro')
f.write("val f1 score:"+str(f1_macro) + ' ')
f.write("val acc score:"+str(f1_micro) + '\n')
print("val f1 score:" + str(f1_macro))
print("val acc score:" + str(f1_micro))
        
pred_train = clf.predict(feasA[ulnodeA])
f1_micro = f1_score(labelsA[ulnodeA], pred_train, average = 'micro')
f1_macro = f1_score(labelsA[ulnodeA], pred_train, average = 'macro')
f.write("test f1 score:"+str(f1_macro) + ' ')
f.write("test acc score:"+str(f1_micro) + '\n')
print("test f1 score:" + str(f1_macro))
print("test acc score:" + str(f1_micro))
        
# A->B
        
clf = SGDClassifier(loss='log', max_iter=1000, random_state=args.seed)
clf.fit(feasA, labelsA)
pred_train = clf.predict(feasA)
f1_micro = f1_score(labelsA, pred_train, average = 'micro')
f1_macro = f1_score(labelsA, pred_train, average = 'macro')
f.write("A->B: ")
f.write("f1 score:"+str(f1_macro) + ' ')
f.write("acc score:"+str(f1_micro) + ' ')
print("f1 score:" + str(f1_macro))
print("acc score:" + str(f1_micro))
        
pred_val = clf.predict(feasB[valnodeB])
f1_micro = f1_score(labelsB[valnodeB], pred_val, average = 'micro')
f1_macro = f1_score(labelsB[valnodeB], pred_val, average = 'macro')
f.write("val f1 score:"+str(f1_macro) + ' ')
f.write("val acc score:"+str(f1_micro) + '\n')
print("val f1 score:" + str(f1_macro))
print("val acc score:" + str(f1_micro))
        
pred_b = clf.predict(feasB[ulnodeB])
f1_micro = f1_score(labelsB[ulnodeB], pred_b, average = 'micro')
f1_macro = f1_score(labelsB[ulnodeB], pred_b, average = 'macro')
f.write("test f1 score:"+str( f1_macro) + ' ')
f.write("test acc score:"+str( f1_micro) + '\n')
print("test f1 score:" + str(f1_macro))
print("test acc score:" + str(f1_micro))


print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

f = open("cite_result/single_embA_{}_lra{}_lrb{}_lamb{}_theta{}_gamma{}_{}.txt".format(args.type,lra,lrb,Lambda,Theta,Gamma,tr),'w')
for item in feasA:
    for i in range(len(item)):
        f.write(str(item[i]))
        f.write(' ')
    f.write('\n')
f = open("cite_result/single_embB_{}_lra{}_lrb{}_lamb{}_theta{}_gamma{}_{}.txt".format(args.type,lra,lrb,Lambda,Theta,Gamma,tr),'w')
for item in feasB:
    for i in range(len(item)):
        f.write(str(item[i]))
        f.write(' ')
    f.write('\n')

#torch.save(model.state_dict(),'nogan_2order_b2net22jjj.pkl')
# Testing
#test()
