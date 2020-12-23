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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler

from utils import load_data, accuracy, F1
from models_m import GCN,NetD,MLP, Center

# cd /mnt/f/Reikun/pku/lab/DANE/pygcn
# python train_multi.py
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=600,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--reconstruct_batch', type=int, default=64,
                    help='batch size for l3')
parser.add_argument('--lra',type=float, default=0.2,
                    help='Ratio of labeled nodes in A.')
parser.add_argument('--lrb',type=float, default=0.2,
                    help='Ratio of labeled nodes in A.')
parser.add_argument('--lamb', type=float, default=5.0,
                    help='Coefficient of GAN.')
parser.add_argument('--gamma', type=float, default=7.0,
                    help='Coefficient of cross_entropy.')
parser.add_argument('--theta', type=float, default=10.0,
                    help='Coefficient of loss_center.')
parser.add_argument('--trial', type=int,default=1,
                    help='Trial times.')
parser.add_argument('--type', type=str, default="DANE",
                    help='Type.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
rbatch_size = args.reconstruct_batch
lra = args.lra
lrb = args.lrb
nA = 1500
nB = 1500
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
Lambda = args.lamb
Theta = args.theta
Gamma = args.gamma
tr = args.trial

one = torch.FloatTensor([1])
mone = one * -1.0
if args.cuda:
    one = one.cuda()
    mone = mone.cuda()
adjA, featuresA, labelsA, idx_trainA, idx_valA, idx_testA, edgesA, edges_weightA, nodes_weightA,multilabelsA = load_data(path="../data/transfer3/", dataset="usa",preserve_order=1)
adjB, featuresB, labelsB, idx_trainB, idx_valB, idx_testB, edgesB, edges_weightB, nodes_weightB,multilabelsB = load_data(path="../data/transfer3/", dataset="chn",preserve_order=1)
nodeA = torch.tensor([1.0 for i in range(0,len(labelsA))])
nodeB = torch.tensor([1.0 for i in range(0,len(labelsB))])
lnodeA = torch.multinomial(nodeA, int(lra*len(labelsA)), replacement=False)
ulnodeA = np.setdiff1d(np.array([i for i in range(len(labelsA))]), lnodeA.numpy())
lnodeB = torch.multinomial(nodeB, int(lrb*len(labelsB)), replacement=False)
ulnodeB = np.setdiff1d(np.array([i for i in range(len(labelsB))]), lnodeB.numpy())
valnodeA = ulnodeA[0:int(0.2*nA)]
valnodeB = ulnodeB[0:int(0.2*nB)]

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

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
                       lr=0.001, weight_decay=args.weight_decay)
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

def cal_cluster(multilabelsA, embA, multilabelsB, embB):
    loss = 0.0
    for i in range(len(multilabelsA[0])):
        idxA = np.where(multilabelsA[:,i] == 0)
        idxB = np.where(multilabelsB[:,i] == 0)
        if (idxA[0].size > 0 and idxB[0].size > 0):
            loss += torch.sum((torch.mean(embA[idxA], dim=0) - torch.mean(embB[idxB], dim = 0))**2)
        idxA = np.where(multilabelsA[:,i] == 1)
        idxB = np.where(multilabelsB[:,i] == 1)
        if (idxA[0].size > 0 and idxB[0].size > 0):
            loss += torch.sum((torch.mean(embA[idxA], dim=0) - torch.mean(embB[idxB], dim = 0))**2)
    return loss/(2*len(multilabelsA[0]))

def loss_semi(multilabel, output):
    multilabel = torch.FloatTensor(multilabel)
    output1 = torch.log(output)
    output1 = torch.mul(multilabel, output1)
    size = torch.numel(output)
    ones = torch.ones(size//4, 4)
    output = torch.log(ones - output)
    output = torch.mul(ones-multilabel, output)
    output = output+output1
    loss = torch.sum(torch.sum(output),dim=0)
    return -loss/size

def MLP_train(epoch_num, featuresrc, featuretgt,labelsrc,labeltgt):
    for i in range(epoch_num):
        idx_trainA = torch.multinomial(nodeA, 140, replacement=True)
        idx_trainB = torch.multinomial(nodeB, 140, replacement=True)
        optimizer_mlp.zero_grad()
        outputsrc = mlp(featuresrc[idx_trainA])
        loss_train = F.nll_loss(outputsrc, labelsrc[idx_trainA])
        outputtgt = mlp(featuretgt[idx_trainB])
        loss_train += F.nll_loss(outputtgt, labeltgt[idx_trainB])
        loss_train.backward(retain_graph = True)
        optimizer_mlp.step()
    
    outputtgt = mlp(featuretgt)
    f1_tgt = F1(outputtgt, labeltgt)
    print("f1 score:"+str(f1_tgt))


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

    loss_semiA = F.nll_loss(outputA[lnodeA], labelsA[lnodeA])
    loss_semiB = F.nll_loss(outputB[lnodeB], labelsB[lnodeB])
    loss_cluster = cal_cluster(multilabelsA[lnodeA], embA[lnodeA], multilabelsB[lnodeB], embB[lnodeB])
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

   
    #loss_train = 10.0*(loss_reconstructionA + loss_reconstructionB + loss_semiA + loss_semiB) + lamb*loss_gan + theta*loss_center
    #loss_train = 10.0 * (loss_reconstructionA + loss_semiA)


    loss_train.backward()
    optimizer.step()
    #print(loss_reconstructionA + loss_reconstructionB)
    #print(loss_gan)




# Train model
f = open("coauthor_result/coauthor_{}_lra{}_lrb{}_lamb{}_theta{}_gamma{}_lr{}_{}.txt".format(args.type,lra,lrb,Lambda,Theta,Gamma,args.lr,tr),'w')
t_total = time.time()

for epoch in range(args.epochs):
    if not (args.type == 'semigsA' or args.type == 'semigsB'):
        for i in range(0,5):
            train_d(epoch)
    if epoch % 100 == 0:
        f.write(str(epoch) + ':\n')
        """
        # B->A
        """
        output, feas = model(featuresB, adjB)
        feas.detach_()
        feas = np.array(feas)
        
        log = MultiOutputClassifier(SGDClassifier(loss='log', max_iter=1000, random_state=args.seed), 4)
        #print(multilabelsB)
        log.fit(feas, multilabelsB) # multiclassifier training on B
        f1_micro = 0.0
        f1_macro = 0.0
        pred_train = log.predict(feas) # test on B
        for i in range(multilabelsB.shape[1]):
            f1_micro += f1_score(multilabelsB[:, i], pred_train[:, i], average='micro')
            f1_macro += f1_score(multilabelsB[:, i], pred_train[:, i], average='macro')
        f1_micro/=4.0
        f1_macro/=4.0
        f.write("B->A: ")
        f.write("f1 score:"+str(f1_macro) + ' ')
        f.write("acc score:"+str(f1_micro) + ' ')
        print("f1 score:" + str(f1_macro))
        print("acc score:" + str(f1_micro))
        
        outputA, feasA = model(featuresA, adjA)
        feasA.detach_()
        feasA = np.array(feasA)
        
        pred_a = log.predict(feasA[valnodeA]) # test on A
        f1_micro = 0.0
        f1_macro = 0.0
        for i in range(multilabelsB.shape[1]):
            f1_micro += f1_score((multilabelsA[valnodeA])[:, i], pred_a[:, i], average='micro')
            f1_macro += f1_score((multilabelsA[valnodeA])[:, i], pred_a[:, i], average='macro')
        f1_micro/=4.0
        f1_macro/=4.0
        f.write("f1 score:"+str( f1_macro) + ' ')
        f.write("acc score:"+str( f1_micro) + '\n')
        print("f1 score:" + str(f1_macro))
        print("acc score:" + str(f1_micro))

        pred_a = log.predict(feasA[ulnodeA])
        f1_micro = 0.0
        f1_macro = 0.0
        for i in range(multilabelsB.shape[1]):
            f1_micro += f1_score((multilabelsA[ulnodeA])[:, i], pred_a[:, i], average='micro')
            f1_macro += f1_score((multilabelsA[ulnodeA])[:, i], pred_a[:, i], average='macro')
        f1_micro/=4.0
        f1_macro/=4.0
        f.write("f1 score:"+str( f1_macro) + ' ')
        f.write("acc score:"+str( f1_micro) + '\n')
        print("f1 score:" + str(f1_macro))
        print("acc score:" + str(f1_micro))
        
        # A->B
        log = MultiOutputClassifier(SGDClassifier(loss='log', max_iter=1000, random_state=args.seed), 4)
        log.fit(feasA, multilabelsA)
        f1_micro = 0.0
        f1_macro = 0.0
        pred_train = log.predict(feasA)
        for i in range(multilabelsB.shape[1]):
            f1_micro += f1_score(multilabelsA[:, i], pred_train[:, i], average='micro')
            f1_macro += f1_score(multilabelsA[:, i], pred_train[:, i], average='macro')
        f1_micro/=4.0
        f1_macro/=4.0
        f.write("A->B: ")
        f.write("f1 score:"+str(f1_macro) + ' ')
        f.write("acc score:"+str(f1_micro) + ' ')
        print("f1 score:" + str(f1_macro))
        print("acc score:" + str(f1_micro))

        pred_b = log.predict(feas[valnodeB])
        f1_micro = 0.0
        f1_macro = 0.0
        for i in range(multilabelsB.shape[1]):
            f1_micro += f1_score((multilabelsB[valnodeB])[:, i], pred_b[:, i], average='micro')
            f1_macro += f1_score((multilabelsB[valnodeB])[:, i], pred_b[:, i], average='macro')
        f1_micro/=4.0
        f1_macro/=4.0
        f.write("f1 score:"+str( f1_macro) + ' ')
        f.write("acc score:"+str( f1_micro) + '\n')
        print("f1 score:" + str(f1_macro))
        print("acc score:" + str(f1_micro))

        pred_b = log.predict(feas[ulnodeB])
        f1_micro = 0.0
        f1_macro = 0.0
        for i in range(multilabelsB.shape[1]):
            f1_micro += f1_score((multilabelsB[ulnodeB])[:, i], pred_b[:, i], average='micro')
            f1_macro += f1_score((multilabelsB[ulnodeB])[:, i], pred_b[:, i], average='macro')
        f1_micro/=4.0
        f1_macro/=4.0
        f.write("f1 score:"+str( f1_macro) + ' ')
        f.write("acc score:"+str( f1_micro) + '\n')
        print("f1 score:" + str(f1_macro))
        print("acc score:" + str(f1_micro))
    #torch.save(model.state_dict(), 'initclear.pkl')
    train_g(epoch,mode = "GAN",lamb=Lambda,theta=Theta, gamma = Gamma)

# final result
f.write("final:\n")
epoch = 0
output, feas = model(featuresB, adjB)
feas.detach_()
feas = np.array(feas)
log = MultiOutputClassifier(SGDClassifier(loss='log', max_iter=1000, random_state=args.seed), 4)
log.fit(feas, multilabelsB)
f1_micro = 0.0
f1_macro = 0.0
pred_train = log.predict(feas)
for i in range(multilabelsB.shape[1]):
    f1_micro += f1_score(multilabelsB[:, i], pred_train[:, i], average='micro')
    f1_macro += f1_score(multilabelsB[:, i], pred_train[:, i], average='macro')
f1_micro/=4.0
f1_macro/=4.0
f.write('B->A')
f.write("f1 score:"+str(f1_macro) + ' ')
f.write("acc score:"+str(f1_micro) + ' ')
print("f1 score:" + str(f1_macro))
print("acc score:" + str(f1_micro))

outputA, feasA = model(featuresA, adjA)
feasA.detach_()
feasA = np.array(feasA)

pred_a = log.predict(feasA[valnodeA])
f1_micro = 0.0
f1_macro = 0.0
for i in range(multilabelsB.shape[1]):
    f1_micro += f1_score((multilabelsA[valnodeA])[:, i], pred_a[:, i], average='micro')
    f1_macro += f1_score((multilabelsA[valnodeA])[:, i], pred_a[:, i], average='macro')
f1_micro/=4.0
f1_macro/=4.0
f.write("f1 score:"+str( f1_macro) + ' ')
f.write("acc score:"+str( f1_micro) + '\n')
print("f1 score:" + str(f1_macro))
print("acc score:" + str(f1_micro))

pred_a = log.predict(feasA[ulnodeA])
f1_micro = 0.0
f1_macro = 0.0
for i in range(multilabelsB.shape[1]):
    f1_micro += f1_score(multilabelsA[ulnodeA][:, i], pred_a[:, i], average='micro')
    f1_macro += f1_score(multilabelsA[ulnodeA][:, i], pred_a[:, i], average='macro')
f1_micro/=4.0
f1_macro/=4.0
f.write("f1 score:"+str( f1_macro) +' ')
f.write("acc score:"+str( f1_micro) + '\n')
print("f1 score:" + str(f1_macro))
print("acc score:" + str(f1_micro))

log = MultiOutputClassifier(SGDClassifier(loss='log', max_iter=1000, random_state=args.seed), 4)
log.fit(feasA, multilabelsA)
f1_micro = 0.0
f1_macro = 0.0
pred_train = log.predict(feasA)
for i in range(multilabelsB.shape[1]):
    f1_micro += f1_score(multilabelsA[:, i], pred_train[:, i], average='micro')
    f1_macro += f1_score(multilabelsA[:, i], pred_train[:, i], average='macro')
f1_micro/=4.0
f1_macro/=4.0
f.write('A->B')
f.write("f1 score:"+str(f1_macro) + ' ')
f.write("acc score:"+str(f1_micro) + ' ')
print("f1 score:" + str(f1_macro))
print("acc score:" + str(f1_micro))

pred_b = log.predict(feas[valnodeB])
f1_micro = 0.0
f1_macro = 0.0
for i in range(multilabelsB.shape[1]):
    f1_micro += f1_score(multilabelsB[valnodeB][:, i], pred_b[:, i], average='micro')
    f1_macro += f1_score(multilabelsB[valnodeB][:, i], pred_b[:, i], average='macro')
f1_micro/=4.0
f1_macro/=4.0
f.write("f1 score:"+str( f1_macro) + ' ')
f.write("acc score:"+str( f1_micro) + '\n')
print("f1 score:" + str(f1_macro))
print("acc score:" + str(f1_micro))

pred_b = log.predict(feas[ulnodeB])
f1_micro = 0.0
f1_macro = 0.0
for i in range(multilabelsB.shape[1]):
    f1_micro += f1_score(multilabelsB[ulnodeB][:, i], pred_b[:, i], average='micro')
    f1_macro += f1_score(multilabelsB[ulnodeB][:, i], pred_b[:, i], average='macro')
f1_micro/=4.0
f1_macro/=4.0
f.write("f1 score:"+str( f1_macro) + ' ')
f.write("acc score:"+str( f1_micro) + '\n')
print("f1 score:" + str(f1_macro))
print("acc score:" + str(f1_micro))

#for epoch in range(200):
    #MLP_train(epoch,feas)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
'''
f = open('./emb/coauthor_embA.txt','w')
for item in feasA:
    for i in range(len(item)):
        f.write(str(item[i]))
        f.write(' ')
    f.write('\n')
f = open('coauthor_labelA.txt','w')
for item in labelsA:
    f.write(str(item))
f = open('./emb/coauthor_embB.txt','w')
for item in feas:
    for i in range(len(item)):
        f.write(str(item[i]))
        f.write(' ')
    f.write('\n')
f = open('coauthor_labelB.txt','w')
for item in labelsB:
    f.write(str(item))
torch.save(model.state_dict(),'nogan_2order_b2net22jjj.pkl')
# Testing
#test()
'''