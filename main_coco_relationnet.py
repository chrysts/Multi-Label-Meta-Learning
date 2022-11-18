"""Training script.
usage: main.py [options]
options:
    --learningrate=lr          Learning rate of inner loop [default: 0.001]
    --dataset=dataset           Dataset [default: coco]
    --datapath=datapath           Datapath [default: /home/csimon/research/data/ms-coco]
    --savepath=savepath           Savepath [default: ./save/coco_rn_resnet]
    --modeltype=model_type     Model Type [default: ResNet]
    --batchsize=bs             Batch size [default: 1]
    --nway=nway                Number of classes [default: 10]
    --shot=shot                Number of shot [default: 1]
    --h, --help                Show help
"""
from docopt import docopt
from dataloader.episode_coco_set_k2 import CocoSet
from utils import pprint, accuracy_calc, process_label_coco_fewshot_definedlabel, Averager, acc_topk_definedlabel, create_mask
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from networks.cnn_encoder_relationet import CNNEncoder
from networks.relation_net import RelationNetwork
from networks.relation_net import RelationNetworkResNet
from EpisodeSamplerMoreLabel import EpisodeSampler
from sklearn.metrics import precision_recall_curve, average_precision_score
import os.path as osp
import os
from networks.cnn import CNN
from torch.nn.utils import clip_grad_norm_
import torchvision.models as models


args = docopt(__doc__)

def load_args( args):
    dd = {}
    dd['lr'] = float(args['--learningrate'])
    dd['dataset'] = str(args['--dataset'])
    dd['datapath'] = str(args['--datapath'])
    dd['savepath'] = str(args['--savepath'])
    dd['modeltype'] = str(args['--modeltype'])
    dd['batchsize'] = int(args['--batchsize'])
    dd['shot'] = int(args['--shot'])
    dd['nway'] = int(args['--nway'])

    return dd

print(args)

pprint(vars(args))

args = load_args(args)

all_label_size = {'coco':81}
total_label_size = args['nway']#all_label_size[args['dataset']]
batch_size = args['batchsize']
num_shot= args['shot']
query_size = total_label_size//2
feature_size = 64
hidden_size= 8
total_traindata = 200
iter  =250

total_label_perepisode = total_label_size

if total_label_size <= 10:
    greaterthan = 0.1
else :
    greaterthan = 1./total_label_size

aug=False
if args['modeltype'] == 'ResNet':
    aug=True

trainset = CocoSet(args['datapath'], 'train', args, aug=False)

train_sampler = EpisodeSampler(ids=trainset.ids, labels=trainset.label, label_ids=trainset.label_ids,
                               query_num=query_size, total_label_size=total_label_size, shot_size=num_shot,
                               n_batch=batch_size, iter=200)
train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)


valset = CocoSet(args['datapath'], 'test', args)

val_sampler = EpisodeSampler(ids=valset.ids, labels=valset.label, label_ids=valset.label_ids,
                               query_num=query_size, total_label_size=total_label_size, shot_size=num_shot,
                               n_batch=batch_size, iter=iter)
val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)

#testset = FlickrSet('test', args)


if args['modeltype'] == 'ConvNet':
    feature_size = 64
    net = CNNEncoder().cuda()
    relationnet = RelationNetwork(feature_size, hidden_size).cuda()
    stepsize = 30

#net = CNN(label_size=total_label_perepisode).cuda()
#net = DynamicCN(models.resnet34(pretrained=True), label_size=label_size)
#label_net = LabelEmbed(label_size=label_size).cuda()

save_path = '-'.join([args['savepath'], args['modeltype'], 'cnn-rnn'])

#if args['model_type'] == 'ConvNet':
optimizer = torch.optim.Adam(list(net.parameters()) + list(relationnet.parameters()), lr=args['lr'])
#optimizer = torch.optim.Adam(net.parameters(), lr=args['lr'])

#optimizer = torch.optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, nesterov=True, weight_decay=0.0001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=0.5)
# elif args['model_type'] == 'ResNet':
#     optimizer = torch.optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, nesterov=True, weight_decay=0.0005)

def save_model(name):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(dict(params=net.state_dict()), osp.join(save_path, name + '.pth'))
    torch.save(dict(params=relationnet.state_dict()), osp.join(save_path, name + '-relation.pth'))

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    net = net.cuda()


maxs_F1 = .0
maxs_PR = .0
maxs_RE = .0

maxs_eval_F1 = 0.0
maxs_eval_PR = 0.0
maxs_eval_RE = 0.0
prototypes = torch.zeros(total_label_perepisode, feature_size, 19, 19).cuda()


for epoch in range(1, 1000):
    lr_scheduler.step()
    cc = 0
    net.train()
    relationnet.train()

    va_PR = Averager()
    va_RE = Averager()
    va_F1 = Averager()
    lossva = Averager()
    loss = 0.0

    for jj, batch in enumerate(train_loader, 0):


        # if epoch == 1 and i == 150 :
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 1e-4
        # if i <= 439:
        #     continue
        data, label_strs = [_ for _ in batch]
        must_label = train_sampler.get_querylabel(jj)
        label_queries, label_support, label_to_idx_support = process_label_coco_fewshot_definedlabel(label_strs, query_size,
                                                                  must_exist_labels=must_label,
                                                                  total_label_episode=total_label_perepisode)

        label_queries = label_queries.cuda().sum(1) #sum over all onehot label for an image
        label_support = label_support.cuda()
        data = data.cuda()

        #prototypes = torch.zeros(total_label_perepisode, feature_size, 19, 19).cuda()

        bolong = []

        ### MAIN ALGORITHM START #####
        prototypes = []#prototypes * 0.
        for key in range(total_label_perepisode):
            # if label_to_idx_support.get(key) == None:
            #     bolong.append(key)
            #     continue
            get_idx = np.array(label_to_idx_support[key])
            support_samples = net(data[get_idx])
            sum_samples = support_samples.sum(dim=0)
            prototypes.append(sum_samples)

        prototypes = torch.stack(prototypes)
        sample_features = prototypes
        batch_features = net(data[:query_size])

        sample_features_ext = sample_features.unsqueeze(0).repeat(query_size,1,1,1,1)
        sample_features_ext = sample_features_ext.reshape(-1, sample_features_ext.shape[-3],
                                                          sample_features_ext.shape[-2],
                                                          sample_features_ext.shape[-1])
        batch_features_ext = batch_features.unsqueeze(1).repeat(1, total_label_perepisode, 1,1,1)
        batch_features_ext = batch_features_ext.reshape(-1, batch_features_ext.shape[-3],
                                                        batch_features_ext.shape[-2],
                                                        batch_features_ext.shape[-1])

        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),1)#.view(-1,feature_size*2,19,19)
        logits = relationnet(relation_pairs).view(-1, total_label_perepisode)



        # mask = create_mask(torch.from_numpy(np.array(bolong)), total_label_perepisode)
        # mask = mask.unsqueeze(0).repeat(query_size, 1).cuda().float()
        # logits = logits*mask

        loss = F.binary_cross_entropy(logits, label_queries.float())#F.mse_loss(logits, label_queries.float())
        ### MAIN ALGORITHM END #####


        acc = accuracy_calc(logits, label_queries.float(), total_label_size, greaterthan=greaterthan, topk=3)
        va_PR.add(acc[0])
        va_RE.add(acc[1])
        va_F1.add(acc[2])
        lossva.add(loss)


        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(net.parameters(), 0.5)
        clip_grad_norm_(relationnet.parameters(), 0.5)
        optimizer.step()
    #
        # print('[TRAIN] epoch {},  loss={:.4f}  avgacc={:.4f} '
        #       .format(epoch, total_traindata//batch_size, loss.item(), va.item()))
    print('[TRAIN] epoch {}'
          .format(epoch))

    if maxs_F1 < va_F1.item():
        maxs_F1 = va_F1.item()


    print('[TRAIN] avg loss={:.4f} PR={:.4f} RE={:.4f} F1={:.4f} maxsF1={:.4f}'
          .format(lossva.item(), va_PR.item(), va_RE.item(), va_F1.item(), maxs_F1))

    save_model('last_epoch-'+ args['modeltype'] +'-coco-relationet-'+str(args['shot'])+'shot')
    if epoch < 150:
        continue



    #### VALIDATION ####################################################################################################
    ap = 0.0
    va_val_F1 = Averager()
    va_val_PR = Averager()
    va_val_RE = Averager()
    lossva_val = Averager()
    acc_all = []
    for jj, batch in enumerate(val_loader, 0):
        net.eval()
        relationnet.eval()
        with torch.no_grad():


            data, label_strs = [_ for _ in batch]
            must_label = val_sampler.get_querylabel(jj)
            label_queries, label_support, label_to_idx_support = process_label_coco_fewshot_definedlabel(label_strs, query_size,
                                                                                                         must_exist_labels=must_label,
                                                                                                         total_label_episode=total_label_perepisode)

            label_queries = label_queries.cuda().sum(1) #sum over all onehot label for an image
            label_support = label_support.cuda()
            data = data.cuda()


            bolong = []
            prototypes = []


            for key in range(total_label_perepisode):
                # if label_to_idx_support.get(key) == None:
                #     bolong.append(key)
                #     continue
                get_idx = np.array(label_to_idx_support[key])
                support_samples = net(data[get_idx])
                sum_samples = support_samples.sum(dim=0)
                prototypes.append(sum_samples)

            prototypes = torch.stack(prototypes)
            sample_features = prototypes
            batch_features_eval = net(data[:query_size])

            sample_features_ext = sample_features.unsqueeze(0).repeat(query_size,1,1,1,1)
            sample_features_ext = sample_features_ext.reshape(-1, sample_features_ext.shape[-3],
                                                              sample_features_ext.shape[-2],
                                                              sample_features_ext.shape[-1])
            batch_features_ext = batch_features_eval.unsqueeze(1).repeat(1, total_label_perepisode, 1,1,1)
            batch_features_ext = batch_features_ext.reshape(-1, batch_features_ext.shape[-3],
                                                            batch_features_ext.shape[-2],
                                                            batch_features_ext.shape[-1])


            relation_pairs = torch.cat((sample_features_ext,batch_features_ext),1)#.view(-1,feature_size*2,19,19)
            logits = relationnet(relation_pairs).view(-1, total_label_perepisode)



            # mask = create_mask(torch.from_numpy(np.array(bolong)), total_label_perepisode)
            # mask = mask.unsqueeze(0).repeat(query_size, 1).cuda().float()
            # logits = logits*mask

            loss = F.mse_loss(logits, label_queries.float())



            acc = accuracy_calc(logits, label_queries.float(), total_label_size, greaterthan=greaterthan, topk=3)
            va_val_PR.add(acc[0])
            va_val_RE.add(acc[1])
            va_val_F1.add(acc[2])
            #acc_all.append(acc[2])
            lossva_val.add(loss)

            logits_np = logits.cpu().numpy()
            label_np = label_queries.cpu().numpy()
            ap_temp = 0.0
            for bb in range(logits_np.shape[0]):
                ap_temp += average_precision_score(label_np[bb], logits_np[bb])
            ap += ap_temp/logits_np.shape[0]
            acc_all.append(ap_temp/logits_np.shape[0])


            # print('epoch {}, eval {}/{}, loss={:.4f} acc={:.4f} avg={:.4f} '
            #       .format(epoch, jj, total_traindata//batch_size, loss.item(), acc, va_val.item()))

    std = np.std(acc_all) * 1.96 / np.sqrt(iter)
    print('Final {}:     F1={:.2f}({:.2f})'.format(epoch, np.mean(acc_all) * 100, std * 100))

    if maxs_eval_F1 < va_val_F1.item():
        maxs_eval_F1 = va_val_F1.item()
        save_model('best_epoch-'+ args['modeltype'] +'-coco-relationet-'+str(args['shot'])+'shot')

    print('[EVAL] avg loss={:.4f} PR={:.4f} RE={:.4f} F1={:.4f} maxsF1={:.4f} map={:.4f}'
          .format(lossva.item(), va_val_PR.item(), va_val_RE.item(), va_val_F1.item(), maxs_eval_F1, ap/iter))