import numpy as np
import scipy.sparse as sp
import torch
import json
import numpy as np
from sklearn import metrics
import math


# each column represents a class
def encode_onehot(labels):
    #classes = set(['AI&DM','SE','CA','CN','CG','CT','HC','other'])
    classes = set(['AI&DM','CA','CN','HC'])
    #classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),dtype=np.int32)
    return labels_onehot


def load_transfer_data():
    ai_venues_conf_a = {'AAAI', 'CVPR', 'ICCV', 'ICML', 'IJCAI', 'NIPS', 'ACL'}
    ai_venues_conf_b = {'COLT', 'EMNLP', 'ECAI', 'ECCV', 'ICRA', 'ICAPS', 'ICCBR',
                        'COLING', 'KR', 'UAI', 'AAMAS', 'PPSN'}
    ai_venues_conf_c = {'ACCV', 'CoNLL', 'GECCO', 'ICTAI', 'ALT', 'ICANN', 'FGR',
                        'ICDAR', 'KSEM', 'ICONIP', 'ICPR', 'Recognition',
                        'ICB', 'IJCNN', 'PRICAI', 'NAACL', 'BMVC', 'IROS', 'AISTATS', 'ACML'}
    ai_venues_peri_a = {'AI', 'TPAMI', 'IJCV', 'JMLR'}
    ai_venues_peri_b = {'TAP', 'TSLP', 'Computational Linguistics', 'CVIU', 'DKE',
                        'Evolutionary Computation', 'TAC', 'TASLP', 'IEEE Transactions on Cybernetics',
                        'TEC', 'TFS', 'TNNLS', 'IJAR', 'JAIR', 'Journal of Automated Reasoning ',
                        'JSLHR', 'AAMAS', 'Machine Learning', 'Neural Computation',
                        'Neural Networks', 'Pattern Recognition'}
    ai_venues_peri_c = {'TALIP', 'Applied Intelligence', 'AIM', 'Artificial Life',
                        'Computational Intelligence', 'Computer Speech and Language',
                        'Connection Science', 'DSS', 'EAAI', 'Expert Systems', 'ESWA',
                        'Fuzzy Sets and Systems', 'T-CIAIG', 'IET-CVI', 'IET Signal Processing',
                        'IVC', 'IDA', 'IJCIA', 'IJDAR', 'IJIS', 'IJNS', 'IJPRAI',
                        'International Journal of Uncertainty, Fuzziness and KBS',
                        'JETAI', 'KBS', 'Machine Translation', 'Machine Vision and Applications',
                        'Natural Computing', 'NLE', 'NCA', 'NPL', 'Neurocomputing',
                        'PAA', 'PRL', 'Soft Computing', 'WIAS'}
    dm_venues_conf_a = {'SIGMOD', 'SIGKDD', 'SIGIR', 'VLDB', 'ICDE'}
    dm_venues_conf_b = {'CIKM', 'PODS', 'DASFAA', 'ECML-PKDD', 'ISWC', 'ICDM', 'ICDT',
                        'EDBT', 'CIDR', 'SDM', 'WSDM'}
    dm_venues_conf_c = {'DEXA', 'ECIR', 'WebDB', 'ER', 'MDM', 'SSDBM', 'WAIM', 'SSTD',
                        'PAKDD', 'APWeb', 'WISE', 'ESWC'}
    dm_venues_peri_a = {'TODS', 'TOIS', 'TKDE', 'VLDBJ'}
    dm_venues_peri_b = {'TKDD', 'AEI', 'DKE', 'DMKD', 'EJIS', 'GeoInformatica', 'IPM',
                        'Information Sciences', 'IS', 'JASIST', 'JWS', 'KIS', 'TWEB'}
    dm_venues_peri_c = {'DPD', 'I&M', 'IPL', 'IR', 'IJCIS', 'IJGIS', 'IJIS', 'IJKM', 'IJSWIS',
                        'JCIS', 'JDM', 'JGITM', 'JIIS', 'JSIS'}

    se_venues_conf_a = {'FSE/ESEC', 'OOPSLA', 'ICSE', 'OSDI', 'PLDI', 'POPL', 'SOSP', 'ASE'}
    se_venues_conf_b = {'ECOOP', 'ETAPS', 'FM', 'ICPC', 'RE', 'CAiSE', 'ICFP', 'LCTES',
                        'MoDELS', 'CP', 'ICSOC', 'ICSME', 'VMCAI', 'ICWS', 'SAS',
                        'ISSRE', 'ISSTA', 'Middleware', 'SANER', 'HotOS', 'ESEM'}
    se_venues_conf_c = {'PASTE', 'APLAS', 'APSEC', 'COMPSAC', 'ICECCS', 'SCAM', 'ICFEM',
                        'TOOLS', 'PEPM', 'QRS', 'SEKE', 'ICSR', 'ICWE', 'SPIN', 'LOPSTR',
                        'TASE', 'ICST', 'ATVA', 'ISPASS', 'SCC', 'ICSSP', 'MSR', 'REFSQ',
                        'WICSA', 'EASE'}
    se_venues_peri_a = {'TOPLAS', 'TOSEM', 'TSE'}
    se_venues_peri_b = {'ASE', 'ESE', 'TSC', 'IETS', 'IST', 'JFP',
                        'Journal of Software: Evolution and Process', 'JSS', 'RE',
                        'SCP', 'SoSyM', 'SPE', 'STVR'}
    se_venues_peri_c = {'CL', 'IJSEKE', 'STTT', 'JLAP', 'JWE', 'SOCA', 'SQJ', 'TPLP'}

    ca_venues_conf_a = {'ASPLOS', 'FAST', 'HPCA', 'ISCA', 'MICRO', 'SC', 'USENIX ATC', 'PPoPP'}
    ca_venues_conf_b = {'HOT CHIPS', 'SPAA', 'PODC', 'CGO', 'DAC', 'DATE', 'EuroSys',
                        'HPDC', 'ICCD', 'ICCAD', 'ICDCS', 'HiPEAC', 'SIGMETRICS', 'ICPP'}
    ca_venues_conf_c = {'CF', 'NOCS', 'ASP-DAC', 'ASAP', 'CLUSTER', 'CCGRID', 'Euro-Par',
                        'ETS', 'FPL', 'FCCM', 'GLSVLSI', 'HPCC', 'MASCOTS', 'NPC', 'ICA3PP',
                        'CASES', 'FPT', 'HPC', 'ICPADS', 'ISCAS', 'ISLPED', 'ISPD',
                        'Hot Interconnects', 'VTS', 'ISPA', 'SYSTOR', 'ATS'}
    ca_venues_peri_a = {'TOCS', 'TOC', 'TPDS', 'TCAD', 'TOS'}
    ca_venues_peri_b = {'TACO', 'TAAS', 'TODAES', 'TECS', 'TRETS', 'TVLSI', 'JPDC',
                        'PARCO', 'Performance Evaluation: An International Journal', 'JSA'}
    ca_venues_peri_c = {'Concurrency and Computation: Practice and Experience ',
                        'DC', 'FGCS', 'Integration',
                        'Microprocessors and Microsystems: Embedded Hardware Design ',
                        'JGC', 'TJSC', 'JETC', 'JET', 'RTS'}

    cn_venues_conf_a = {'MOBICOM', 'SIGCOMM', 'INFOCOM'}
    cn_venues_conf_b = {'SenSys', 'CoNEXT', 'SECON', 'IPSN', 'ICNP', 'MobiHoc', 'MobiSys',
                        'IWQoS', 'IMC', 'NOSSDAV', 'NSDI'}
    cn_venues_conf_c = {'ANCS', 'FORTE', 'LCN', 'Globecom', 'ICC', 'ICCCN', 'MASS', 'P2P',
                        'IPCCC', 'WoWMoM', 'ISCC', 'WCNC', 'Networking', 'IM', 'MSWiM',
                        'NOMS', 'HotNets', 'WASA'}
    cn_venues_peri_a = {'TON', 'JSAC', 'TMC'}
    cn_venues_peri_b = {'TOIT', 'TOMCCAP', 'TOSN', 'CN', 'TOC', 'TWC'}
    cn_venues_peri_c = {'Ad hoc Networks', 'CC', 'TNSM', 'IET Communications', 'JNCA',
                        'MONET', 'Networks', 'PPNA', 'WCMC', 'Wireless Networks'}

    cg_venues_conf_a = {'ACM MM', 'SIGGRAPH', 'IEEE VIS', 'VR'}
    cg_venues_conf_b = {'ICMR', 'i3D', 'SCA', 'DCC', 'EG', 'EuroVis', 'SGP', 'EGSR',
                        'ICME', 'PG', 'SPM', 'ICASSP'}
    cg_venues_conf_c = {'CASA', 'CGI', 'ISMAR', 'PacificVis', 'ICIP', 'MMM', 'GMP', 'PCM',
                        'SMI', 'INTERSPEECH',
                        'ACM Symposium on Virtual Reality Software and Technology'}
    cg_venues_peri_a = {'TOG', 'TIP', 'TVCG'}
    cg_venues_peri_b = {'TOMCCAP', 'CAD', 'CAGD', 'CGF', 'GM', 'TCSVT', 'TMM', 'JASA',
                        'SIIMS', 'Speech Com'}
    cg_venues_peri_c = {'CAVW', 'C&G', 'CGTA', 'DCG', 'IET-IPR', 'IEEE Signal Processing Letter',
                        'JVCIR', 'MS', 'MTA', 'Signal Processing', 'SPIC', 'TVC'}

    ct_venues_conf_a = {'STOC', 'FOCS', 'LICS', 'CAV'}
    ct_venues_conf_b = {'SoCG', 'SODA', 'CADE/IJCAR', 'CCC', 'ICALP', 'CONCUR', 'HSCC', 'ESA'}
    ct_venues_conf_c = {'CSL', 'FSTTCS', 'IPCO', 'RTA', 'ISAAC', 'MFCS', 'STACS', 'FMCAD',
                        'SAT', 'ICTAC'}
    ct_venues_peri_a = {'IANDC', 'SICOMP', 'TIT'}
    ct_venues_peri_b = {'TALG', 'TOCL', 'TOMS', 'Algorithmica', 'CC', 'FAC', 'FMSD',
                        'INFORMS', 'JCSS', 'JGO', 'JSC', 'MSCS', 'TCS'}
    ct_venues_peri_c = {'APAL', 'ACTA', 'DAM', 'FUIN', 'LISP', 'JCOMPLEXITY',
                        'LOGCOM', 'Journal of Symbolic Logic', 'LMCS', 'SIDMA',
                        'Theory of Computing Systems'}

    hc_venues_conf_a = {'CHI', 'UbiComp', 'CSCW'}
    hc_venues_conf_b = {'IUI', 'ITS', 'UIST', 'ECSCW', 'MobileHCI', 'PERCOM', 'GROUP'}
    hc_venues_conf_c = {'ASSETS', 'DIS', 'GI', 'MobiQuitous', 'INTERACT', 'CoopIS', 'ICMI',
                        'IDC', 'AVI', 'UIC', 'DIS', 'IEEE World Haptics Conference',
                        'CSCWD', 'CollaborateCom'}
    hc_venues_peri_a = {'TOCHI', 'IJHCS'}
    hc_venues_peri_b = {'CSCW', 'HCI', 'IWC', 'UMUAI', 'IJHCI',
                        'IEEE Transactions on Human-Machine Systems'}
    hc_venues_peri_c = {'BIT', 'PMC', 'PUC'}

    ai_dm_venues = ai_venues_conf_a | ai_venues_conf_b | ai_venues_conf_c | \
                   ai_venues_peri_a | ai_venues_peri_b | ai_venues_peri_c | \
                   dm_venues_conf_a | dm_venues_conf_b | dm_venues_conf_c | \
                   dm_venues_peri_a | dm_venues_peri_b | dm_venues_peri_c

    software_engineering_venues = se_venues_conf_b | se_venues_conf_b | se_venues_conf_c | \
                                  se_venues_peri_a | se_venues_peri_b | se_venues_peri_c

    computer_architecture_venues = ca_venues_conf_a | ca_venues_conf_b | ca_venues_conf_c | \
                                   ca_venues_peri_a | ca_venues_peri_b | ca_venues_peri_c

    computer_network_venues = cn_venues_conf_a | cn_venues_conf_b | cn_venues_conf_c | \
                              cn_venues_peri_a | cn_venues_peri_b | cn_venues_peri_c

    computer_graphics_venues = cg_venues_conf_a | cg_venues_conf_b | cg_venues_conf_c | \
                               cg_venues_peri_a | cg_venues_peri_b | cg_venues_peri_c

    computer_theory_venues = ct_venues_conf_a | ct_venues_conf_b | ct_venues_conf_c | \
                             ct_venues_peri_a | ct_venues_peri_b | ct_venues_peri_c

    human_computer_interaction_venues = hc_venues_conf_a | hc_venues_conf_b | hc_venues_conf_c | \
                                        hc_venues_peri_a | hc_venues_peri_b | hc_venues_peri_c

    all_venues = ai_dm_venues | software_engineering_venues | \
                 computer_architecture_venues | computer_network_venues | \
                 computer_graphics_venues | computer_theory_venues | \
                 human_computer_interaction_venues

    VENUE_DIRECTORY = {'AI&DM': ai_dm_venues,
                       'SE': software_engineering_venues,
                       'CA': computer_architecture_venues,
                       'CN': computer_network_venues,
                       'CG': computer_graphics_venues,
                       'CT': computer_theory_venues,
                       'HC': human_computer_interaction_venues,
                       'ALL': all_venues}
    path = "../data/transfer/"
    dataset = "china_result.txt"
    f = open(path+dataset)
    chn_paper = []
    for line in f:
        chn_paper += json.loads(line)
    dataset = "usa_result.txt"
    f = open(path + dataset)
    usa_paper = []
    for line in f:
        usa_paper += json.loads(line)


def load_data(path="../data/transfer/", dataset="chn", preserve_order=1):
    """Load citation network dataset (cora only for now)"""

    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1]) # labels are at the end of each line
    #f = open("{}{}.multilabel".format(path, dataset))
    #multilabels =np.genfromtxt("{}{}.multilabel".format(path, dataset),
    #                           dtype=np.dtype(str))

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = sp.coo_matrix(adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj))
    for item in adj.__dict__.items():
        print(item)
    print(adj.col)

    edge_ret = []

    edge_weight = []

    node_weight = [0.0 for i in range(0, len(idx))]

    if preserve_order == 1:
        adj_pres = adj
    else:
        adj_pres = sp.coo_matrix(adj**2)

    # sampling weight
    for i in range(0, len(adj.data)):
        edge_ret.append((adj_pres.row[i], adj_pres.col[i]))
        edge_weight.append(float(adj_pres.data[i]))
        node_weight[adj.row[i]] += adj.data[i]


    features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])
    D = sp.coo_matrix([[1.0/math.sqrt(node_weight[j]) if j== i else 0 for j in range(len(idx))] for i in range(len(idx))])
    adj = D*adj*D

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    for i in range(0, len(node_weight)):
        node_weight[i] = math.pow(node_weight[i],0.75)

    return adj, features, labels, idx_train, idx_val, idx_test, edge_ret, torch.tensor(edge_weight), torch.tensor(node_weight)#, multilabels


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    #correct = preds.eq(labels).double()
    #correct = correct.sum()
    #return correct / len(labels)
    return metrics.f1_score(labels, preds, average='micro')

def F1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    return metrics.f1_score(labels, preds, average='macro')


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
