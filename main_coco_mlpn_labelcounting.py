

"""Training script.
usage: main.py [options]
options:
    --learningrate=lr          Learning rate of inner loop [default: 0.001]
    --dataset=dataset           Dataset [default: coco]
    --datapath=datapath           Datapath [default: /home/csimon/research/data/ms-coco]
    --savepath=savepath           Savepath [default: ./save/imaterialist_mlpn]
    --modeltype=model_type     Model Type [default: ConvNet]
    --batchsize=bs             Batch size [default: 1]
    --nway=nway                Number of classes [default: 10]
    --shot=shot                Number of shot [default: 1]
    --h, --help                Show help
"""
from docopt import docopt
from dataloader.episode_coco_set_k2 import CocoSet
from utils import pprint, accuracy_calc, process_label_coco_fewshot_definedlabel, Averager, count_acc_onehot, count_acc
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from networks.DCN import DynamicCN
from networks.label_embed import LabelEmbed
from networks.cnn_encoder_relationet import CNNEncoder
from networks.backbone import ResNet10
from networks.convnet import ConvNet
from EpisodeSamplerMoreLabel import EpisodeSampler
from sklearn.metrics import precision_recall_curve, average_precision_score
import os.path as osp
import os
from networks.cnn import CNN
from torch.nn.utils import clip_grad_norm_
import torchvision.models as models
from networks.selfsupervision import LabelCounter


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
feature_size = 1600
hidden_size= 8
total_traindata = 200
#sigma = 50
iter = 250
greaterthan = 0.1

total_label_perepisode = total_label_size

aug=False
if args['modeltype'] == 'ResNet':
    aug=True

trainset = CocoSet(args['datapath'], 'train', args, aug=aug)

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
    net = ConvNet().cuda()
    stepsize = 30
    #sigma = 50
    sigma = 50
    label_counter = LabelCounter(feature_dim=1600).cuda()
else:
    net = ResNet10().cuda()
    stepsize = 120
    sigma = 30.
    label_counter = LabelCounter(feature_dim=512).cuda()

save_path = '-'.join([args['savepath'], args['modeltype'], 'cnn-rnn'])

#if args['model_type'] == 'ConvNet':
optimizer = torch.optim.Adam(list(net.parameters())+list(label_counter.parameters()), lr=args['lr'])


lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=0.5)
# elif args['model_type'] == 'ResNet':
#     optimizer = torch.optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, nesterov=True, weight_decay=0.0005)

def save_model(name):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(dict(params=net.state_dict()), osp.join(save_path, name + '.pth'))

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    net = net.cuda()


maxs_train_F1 = .0
maxs_eval_F1 = 0.0
prototypes = torch.zeros(total_label_perepisode, feature_size).cuda()


for epoch in range(1, 1000):
    lr_scheduler.step()
    cc = 0
    net.train()

    va_PR = Averager()
    va_RE = Averager()
    va_F1 = Averager()
    lossva = Averager()
    loss = 0.0

    va_num = Averager()
    absolute_acc_count = Averager()

    for jj, batch in enumerate(train_loader, 0):

        #jj = 0
        data, label_strs = [_ for _ in batch]
        must_label = train_sampler.get_querylabel(jj)
        label_queries, label_support, label_to_idx_support = process_label_coco_fewshot_definedlabel(label_strs, query_size,
                                                                                                     must_exist_labels=must_label,
                                                                                                     total_label_episode=total_label_perepisode)

        label_queries = label_queries.cuda().sum(1) #sum over all onehot label for an image
        label_support = label_support.cuda().sum(1)
        label_support_num = label_support.sum(1)
        label_queries_num = label_queries.sum(1)

        label_support_num_onehot = torch.zeros(label_support_num.shape[0], total_label_perepisode).cuda()
        label_support_num_onehot.fill_(0.)
        label_support_num_onehot.scatter_(1, label_support_num.data.unsqueeze(1), 1.)

        label_query_num_onehot = torch.zeros(label_queries_num.shape[0], total_label_perepisode).cuda()
        label_query_num_onehot.fill_(0.)
        label_query_num_onehot.scatter_(1, label_queries_num.data.unsqueeze(1), 1.)

        data = data.cuda()

        all_label_init = torch.cat((label_queries*0.0 + 1e-12, label_support), dim=0)
        all_label = torch.cat((label_support, label_queries), dim=0)

        all_sample_size = data.shape[0]
        all_samples = net(data)

        support_samples = all_samples[query_size:]
        query_features = net(data[:query_size])

        all_samples_arranged = torch.cat((support_samples, query_features), dim=0)


        idxs_to_nol = []
        array_sample_num = np.array([range(all_sample_size)])
        mask = (torch.ones(all_sample_size, all_sample_size) - torch.eye(all_sample_size, all_sample_size)).cuda().double()

        if num_shot > 1:
            for i_label in range(all_sample_size - query_size):
                selected_idxs = np.reshape(np.argwhere(all_label[i_label].cpu().numpy()), -1)
                get_idx = np.array([])
                for key in selected_idxs:
                    get_idx = np.concatenate((get_idx, np.array(label_to_idx_support[key])), axis=0)

                get_idx = get_idx.astype(int)
                mask[i_label, array_sample_num] = 0
                mask[i_label, get_idx] = 1

        s1 = all_samples_arranged.unsqueeze(1).repeat(1, all_samples_arranged.shape[0], 1)
        s2 = all_samples_arranged.unsqueeze(0).repeat(all_samples_arranged.shape[0], 1, 1)
        dist_all = torch.sum((s1 - s2)**2, dim=-1)/sigma
        dist_all = torch.exp(-dist_all.double())
        dist_all = dist_all*mask
        dist_all = dist_all/torch.sum(dist_all, dim=-1).unsqueeze(-1).repeat(1, dist_all.shape[0])

        identity = torch.ones(dist_all.shape[0]).cuda().double()
        identity = identity.diag()

        AA = (identity - dist_all)

        A_ul = AA[-query_size:, :-query_size]
        A_uu = AA[-query_size:, -query_size:]
        identity_query = torch.ones(A_uu.shape[0]).cuda().double()
        identity_query = identity_query.diag()
        A_uu_inv = (A_uu).inverse()

        label_support_norm = label_support.float()/torch.norm(label_support.float(), p=1, dim=-1, keepdim=True)
        label_queries_norm = label_queries.float()/torch.norm(label_queries.float(), p=1, dim=-1, keepdim=True)


        FF = -torch.mm(A_uu_inv, torch.mm(A_ul, label_support_norm.double()))
        #FF = F.relu(-torch.mm(A_ul, label_support_norm.double()))
        logits = FF


        all_support_features = net(data[query_size:])
        results, y_pred, _ = label_counter(all_support_features, query_features, label_support_num)
        gt_count_label_ori = torch.cat((label_queries_num, label_support_num), dim=0)
        gt_count_label = label_queries_num.unsqueeze(1) + label_support_num.unsqueeze(0)

        loss_count = F.cross_entropy(results.view(-1, results.shape[-1]), gt_count_label.view(-1).long())

        logits = F.softmax(logits, dim=-1)
        logits_clone = logits.clone()

        loss = F.mse_loss(logits.float(), label_queries_norm.float()) + 0.01*loss_count

        y_pred_hard_constr = y_pred.clone()
        y_pred_hard_constr = y_pred_hard_constr.long().cuda()
        y_pred_hard_constr[y_pred_hard_constr < 0] = 0
        y_pred_hard_constr[y_pred_hard_constr > total_label_size - 1] = 0
        y_pred_onehot = torch.zeros(label_queries_num.shape[0], total_label_perepisode*2).cuda()
        y_pred_onehot.fill_(0.)
        y_pred_onehot.scatter_(1, y_pred_hard_constr.data.unsqueeze(1), 1.)


        absolute_acc = count_acc_onehot(logits, label_queries, topkpred=y_pred_onehot)
        acc = accuracy_calc(logits, label_queries.float(), label_size=total_label_size, greaterthan=greaterthan)
        acc_num = count_acc(y_pred_onehot, label_queries_num)#.float(), label_size=total_label_size, greaterthan=greaterthan)

        va_num.add(acc_num)
        absolute_acc_count.add(absolute_acc)

        va_PR.add(acc[0])
        va_RE.add(acc[1])
        va_F1.add(acc[2])
   
        lossva.add(loss)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('[TRAIN] epoch {}'
          .format(epoch))

    if maxs_train_F1 < va_F1.item():
        maxs_train_F1 = va_F1.item()


    print('[TRAIN] avg loss={:.4f} PR={:.4f} RE={:.4f} F1={:.4f} maxsF1={:.4f}, va_num={:.4f}, absolute_acc_count={:.4f}'
           .format(lossva.item(), va_PR.item(), va_RE.item(), va_F1.item(), maxs_train_F1, va_num.item(), absolute_acc_count.item()))


    save_model('last_epoch-'+ args['modeltype'] +'-coco-trans-'+str(args['nway'])+'-'+str(args['shot'])+'shot')
    if epoch < 150:
        continue


    va_val_F1 = Averager()
    va_val_PR = Averager()
    va_val_RE = Averager()
    lossva_val = Averager()
    acc_all = []
    va_num = Averager()
    absolute_acc_count = Averager()
    ap = 0.0

    for jj, batch in enumerate(val_loader, 0):
        net.eval()
        with torch.no_grad():


            data, label_strs = [_ for _ in batch]
            must_label = val_sampler.get_querylabel(jj)
            label_queries, label_support, label_to_idx_support = process_label_coco_fewshot_definedlabel(label_strs, query_size,
                                                                                                         must_exist_labels=must_label,
                                                                                                         total_label_episode=total_label_perepisode)

            label_queries = label_queries.cuda().sum(1) #sum over all onehot label for an image
            label_support = label_support.cuda().sum(1)
            label_support_num = label_support.sum(1)
            label_queries_num = label_queries.sum(1)

            label_support_num_onehot = torch.zeros(label_support_num.shape[0], total_label_perepisode).cuda()
            label_support_num_onehot.fill_(0.)
            label_support_num_onehot.scatter_(1, label_support_num.data.unsqueeze(1), 1.)

            label_query_num_onehot = torch.zeros(label_queries_num.shape[0], total_label_perepisode).cuda()
            label_query_num_onehot.fill_(0.)
            label_query_num_onehot.scatter_(1, label_queries_num.data.unsqueeze(1), 1.)


            data = data.cuda()

            all_label_init = torch.cat((label_queries*0.0 + 1e-12, label_support), dim=0)
            all_label = torch.cat((label_support, label_queries), dim=0)

            all_sample_size = data.shape[0]
            all_samples = net(data)

            support_samples = all_samples[query_size:]
            query_features = net(data[:query_size])

            all_samples_arranged = torch.cat((support_samples, query_features), dim=0)


            idxs_to_nol = []
            array_sample_num = np.array([range(all_sample_size)])
            mask = (torch.ones(all_sample_size, all_sample_size) - torch.eye(all_sample_size, all_sample_size)).cuda().double()

            if num_shot > 1:
                for i_label in range(all_sample_size - query_size):
                    selected_idxs = np.reshape(np.argwhere(all_label[i_label].cpu().numpy()), -1)
                    get_idx = np.array([])
                    for key in selected_idxs:
                        get_idx = np.concatenate((get_idx, np.array(label_to_idx_support[key])), axis=0)

                    get_idx = get_idx.astype(int)
                    mask[i_label, array_sample_num] = 0
                    mask[i_label, get_idx] = 1

            s1 = all_samples_arranged.unsqueeze(1).repeat(1, all_samples_arranged.shape[0], 1)
            s2 = all_samples_arranged.unsqueeze(0).repeat(all_samples_arranged.shape[0], 1, 1)
            dist_all = torch.sum((s1 - s2)**2, dim=-1)/sigma
            dist_all = torch.exp(-dist_all.double())
            dist_all = dist_all*mask
            dist_all = dist_all/torch.sum(dist_all, dim=-1).unsqueeze(-1).repeat(1, dist_all.shape[0])

            identity = torch.ones(dist_all.shape[0]).cuda().double()
            identity = identity.diag()

            AA = (identity - dist_all)

            A_ul = AA[-query_size:, :-query_size]
            A_uu = AA[-query_size:, -query_size:]
            identity_query = torch.ones(A_uu.shape[0]).cuda().double()
            identity_query = identity_query.diag()
            A_uu_inv = (A_uu).inverse()

            label_support_norm = label_support.float()/torch.norm(label_support.float(), p=1, dim=-1, keepdim=True)
            label_queries_norm = label_queries.float()/torch.norm(label_queries.float(), p=1, dim=-1, keepdim=True)


            FF = F.relu(-torch.mm(A_ul, label_support_norm.double()))
            logits = FF
            loss = F.mse_loss(logits.float(), label_queries_norm) #+ protoloss

            all_support_features = net(data[query_size:])
            results, y_pred, num_label_ori_pred = label_counter(all_support_features, query_features, label_support_num)
            gt_count_label_ori = torch.cat((label_queries_num, label_support_num), dim=0)
            gt_count_label = label_queries_num.unsqueeze(1) + label_support_num.unsqueeze(0)

            loss_count = F.cross_entropy(results.view(-1, results.shape[-1]), gt_count_label.view(-1).long())

            logits = F.softmax(logits, dim=-1)
            logits_clone = logits.clone()

            y_pred_hard_constr = y_pred.clone()
            y_pred_hard_constr = y_pred_hard_constr.long().cuda()
            y_pred_hard_constr[y_pred_hard_constr < 0] = 0
            y_pred_hard_constr[y_pred_hard_constr > total_label_size -1] = 0
            y_pred_onehot = torch.zeros(label_queries_num.shape[0], total_label_perepisode*2).cuda()
            y_pred_onehot.fill_(0.)
            y_pred_onehot.scatter_(1, y_pred_hard_constr.data.unsqueeze(1), 1.)

            absolute_acc = count_acc_onehot(logits, label_queries, topkpred=y_pred_onehot)
            acc = accuracy_calc(logits, label_queries.float(), label_size=total_label_size, greaterthan=0.1)
            acc_num = count_acc(y_pred_onehot, label_queries_num)#.float(), label_size=total_label_size, greaterthan=greaterthan)

            va_num.add(acc_num)
            absolute_acc_count.add(absolute_acc)


            va_val_PR.add(acc[0])
            va_val_RE.add(acc[1])
            va_val_F1.add(acc[2])
            lossva_val.add(loss)

            logits_np = logits.cpu().numpy()
            label_np = label_queries.cpu().numpy()
            ap_temp = 0.0
            for bb in range(logits_np.shape[0]):
                ap_temp += average_precision_score(label_np[bb], logits_np[bb])
            ap += ap_temp/logits_np.shape[0]
            acc_all.append(ap_temp/logits_np.shape[0])

            
    std = np.std(acc_all) * 1.96 / np.sqrt(iter)
    print('Final {}: {:.2f}({:.2f})'.format(epoch, np.mean(acc_all) * 100, std * 100))

    if maxs_eval_F1 < va_val_F1.item():
        maxs_eval_F1 = va_val_F1.item()
        save_model('best_epoch-'+ args['modeltype'] +'-coco-trans-woquery-'+str(args['shot'])+'shot')

    print('[EVAL] avg loss={:.4f} PR={:.4f} RE={:.4f} F1={:.4f} maxsF1={:.4f} map={:.4f}, va-num={:.4f}, absolute_acc_count={:.4f}'
          .format(lossva.item(), va_val_PR.item(), va_val_RE.item(), va_val_F1.item(), maxs_eval_F1, ap/iter, va_num.item(), absolute_acc_count.item()))
