import os
import shutil
import time
import pprint
import torch
import numpy as np

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def nullisasi(logits, pred_topk):
    try:
        logits_copy = logits.clone().detach().cpu()
    except:
        print("error")
    for jj in range(logits_copy.shape[0]):
        topknum = torch.argmax(pred_topk[jj]).int()
        if topknum < 1:
            topknum = 1
        #topknum = torch.clamp(topknum, 1)
        #if topknum < 1:
        #    topknum = 1
        try:
            topkval, idxx = torch.topk(logits_copy[jj], k=topknum, dim=-1)
        except:
            print("topk: ")
            print(topknum)
       # idxx = idxx.detach().cpu().numpy()
        logits_copy[jj, idxx] = 1.

    logits_copy[logits_copy< 1.] = 0.

    return logits_copy


def count_acc_onehot(logits, label_onehot, topkpred):
    #pred = torch.argmax(logits, dim=1)
    newlogits = nullisasi(logits, topkpred)
    a = newlogits.detach().cpu().numpy()
    b = label_onehot.cpu().numpy()
    #if torch.cuda.is_available():
    return np.mean(np.all(np.equal(a, b), axis=1))#(logits == label_onehot).type(torch.cuda.FloatTensor).mean().item()
    #else:
    #    return (logits == label_onehot).type(torch.FloatTensor).mean().item()


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def process_label_coco(label_strs, all_label_size):
    pad_token = 0
    labels = []

    for ii in range(len(label_strs)):
        #
        candidate = [int(b) + 1 for b in label_strs[ii].split(',')] # labels should start from 1
        labels.append(candidate)


    label_lengths = [len(label) for label in labels]
    maks = max(label_lengths)
    if maks > 15:
        maks = 15

    padded_labels = np.ones((len(label_strs), maks)) * pad_token

    for i, x_len in enumerate(label_lengths):
        if x_len  > maks:
            x_len = maks
        sequence = labels[i]
        padded_labels[i, 0:x_len] = sequence[:x_len]

    # sort based on sequence

    #masks = (padded_labels > 0) *1
    #label_size_perow = np.sum(masks, axis=-1)
    #masks = np.repeat(np.expand_dims(masks, axis=-1), all_label_size, axis=-1)
    label_lengths = torch.from_numpy(np.array(label_lengths)).cuda()
    label_lengths, perm_idx = label_lengths.sort(0, descending=True)
    padded_labels = torch.from_numpy(padded_labels[perm_idx]).cuda().long()

    return padded_labels, label_lengths#, masks, label_size_perow


def process_label(label_strs, all_label_size):
    pad_token = 0
    labels = []
    #14, 20, 15, 19, 10, 7, 12, 23, 16, 5, 24, 1, 22, 13, 21, 8, 18, 4, 17, 9, 11, 3, 6, 2
    label_order = {14:0, 20:1, 15:2, 19:3, 10:4, 7:5, 12:6, 23:7, 16:8, 5:9, 24:10, 1:11, 22:12, 13:13, 21:14, 8:15,
                   18:16, 4:17, 17:18, 9:19, 11:20, 3:21, 6:22, 2:23}
    label_reverseorder = {0:14, 1:20, 2:15, 3:19, 4:10, 5:7, 6:12, 7:23, 8:16, 9:5, 10:24, 11:1, 12:22, 13:13, 14:21, 15:8,
                          16:18, 17:4, 18:17, 19:9, 20:11, 21:3, 22:6, 23:2}

    for ii in range(len(label_strs)):
        #
        candidate = [label_order[int(b)] for b in label_strs[ii].split(',')]
        sort = sorted(candidate)
        labels.append([label_reverseorder[b] for b in sort])


    label_lengths = [len(label) for label in labels]
    maks = max(label_lengths)
    if maks > 15:
        maks = 15

    padded_labels = np.ones((len(label_strs), maks)) * pad_token

    for i, x_len in enumerate(label_lengths):
        if x_len  > maks:
            x_len = maks
        sequence = labels[i]
        padded_labels[i, 0:x_len] = sequence[:x_len]

    # sort based on sequence 
    label_lengths = torch.from_numpy(np.array(label_lengths)).cuda()
    label_lengths, perm_idx = label_lengths.sort(0, descending=True)
    padded_labels = torch.from_numpy(padded_labels[perm_idx]).cuda().long()

    return padded_labels, label_lengths#, masks, label_size_perow


def count_acc_multilabel_rnn(logits, label, label_size=25, batch_size=32):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():

        label_onehot = one_hot(label.view(-1, 3), label_size).sum(1)
        pred_onehot = one_hot(pred.view(-1, 3), label_size).sum(1)


        all = torch.sum(pred_onehot*label_onehot).float()/(3*batch_size) 
        return all
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def count_acc_multilabel(logits, label, batch_size=32):
    if torch.cuda.is_available():
        pred = logits.clone()
        _, perm_idx_pred = pred.sort(-1, descending=True)
        perm_idx_pred = perm_idx_pred[:, :3].sort(-1)
        perm_idx_pred = perm_idx_pred[0]
        pred = pred * 0.0
        for i in range(batch_size):
            for j in range(3):
                pred[i, perm_idx_pred[i, j]] = 1.0


        
        #
        all = torch.sum(pred * label)/(3*batch_size)
        count = 0.0
 
        return all
    else:
        return torch.mean(logits*label)


def one_hot_1d(label, label_size):
    label = torch.Tensor([label]).cuda().long()
    one_hot_labels = torch.zeros_like(label)
    temp_lbl = label.unsqueeze(-1)
    one_hot_labels = one_hot_labels.unsqueeze(-1).repeat(1, label_size)
    one_hot_labels.scatter_(1, temp_lbl.data, 1)

    return torch.squeeze(one_hot_labels).float()

def one_hot(labels, label_size):
    one_hot_labels = torch.zeros_like(labels)
    temp_lbl = labels.unsqueeze(-1)
    one_hot_labels = one_hot_labels.unsqueeze(-1).repeat(1, 1, label_size)
    one_hot_labels.scatter_(2, temp_lbl.data, 1)

    return one_hot_labels


def one_hot_all(labels, label_size):
    shape_0, shape_1 = labels.shape
    labelsflatten = labels.reshape(-1)
    one_hot_labels = torch.zeros_like(labelsflatten)
    temp_lbl = labelsflatten.unsqueeze(-1)
    one_hot_labels = one_hot_labels.unsqueeze(-1).repeat(1, label_size+1)
    one_hot_labels.scatter_(1, temp_lbl.data, 1)
    one_hot_labels = torch.squeeze(one_hot_labels).float()
    one_hot_labels = one_hot_labels.reshape(shape_0, shape_1, -1)

    return one_hot_labels[:, :, :-1]

def process_label_coco_fewshot(label_strs, num_query, num_shot, num_labelperimage, must_exist_labels, total_label_episode):
    total_label = total_label_episode
    labels = []
    labelmap = {}
    lbl_count = 0
    label_to_idx = {}

    label_count = {}
    query_label = {}

    for jj in range(len(must_exist_labels)):
        query_label[must_exist_labels[jj]] = 1


    for ii in range(len(label_strs)):
        #
        candidates = [int(b) for b in label_strs[ii].split(',')]
        priority_candidates = []
        else_candidates = []
        for candidate in candidates:
            if query_label.get(candidate) != None:
                priority_candidates.append(candidate)
            else :
                else_candidates.append(candidate)
            # if labelmap.get(candidate) == None:
            #     labelmap[candidate] = lbl_count
            #     lbl_count += 1
            if label_count.get(candidate) == None:
                label_count[candidate] = 0
            else:
                label_count[candidate] += 1

        priority_candidates = np.array(priority_candidates)
        else_candidates = np.array(else_candidates)
        final_candidates = np.concatenate((priority_candidates, else_candidates), axis=0)
        labels.append(final_candidates)

   


    random_label = torch.randperm(total_label) + 1 # 1 base
    relabel = []#labels.clone()
    all_labels = labels#.view(-1)


    #RELABEL and LABEL to IDXS mapping
    for jj in range(len(all_labels)):
        new_label = []
        cc = 0
        for ii in all_labels[jj]:
            if cc > 2:
                break
            if labelmap.get(ii) == None:
                labelmap[ii] = lbl_count
                lbl_count += 1
                # if lbl_count > 26:
                #     print(labelmap)
                #     print(len(all_labels))
            new_label.append(random_label[labelmap[ii]]) #back to 0 base
            cc += 1

        new_labels = np.array(new_label)[:num_labelperimage]
        if jj > num_query-1: #indexing is just for support set
            for new_label in new_labels:
                nlbl_idxzero = new_label - 1
                if label_to_idx.get(nlbl_idxzero) == None:
                    label_to_idx[nlbl_idxzero] = []
                label_to_idx[nlbl_idxzero].append(jj) #with query and support set
        relabel.append(new_labels)

    padded_labels = pad_labels(relabel, maks=3)

    new_label_onehot = one_hot(torch.from_numpy(padded_labels).long(), total_label+1)

    new_label_onehot_cutzero = new_label_onehot[:, :, 1:]

    return new_label_onehot_cutzero[:num_query], new_label_onehot_cutzero[num_query:], label_to_idx


def pad_labels(labels, maks):
    pad_token = 0
    padded_labels = np.ones((len(labels), maks)) * pad_token

    for i, x_len in enumerate(labels):
        tmp = len(x_len)
        if len(x_len)  > maks:
            tmp = maks
        sequence = labels[i]
        padded_labels[i, 0:tmp] = sequence[:tmp]

    return padded_labels


#def acc_topk(input, target, k=3):
def acc_topk(logits, label, k=3):
    label_size = logits.shape[-1]
    sample_size = logits.shape[0]
    _, pred = torch.topk(logits, k=k, dim=-1)
    if torch.cuda.is_available():
 
        pred_onehot = one_hot(pred, label_size).sum(1)
        #

        all = torch.sum(pred_onehot*label).float()/(k*sample_size) 
        return all
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()





def process_label_coco_fewshot_limitedlabel(label_strs, num_query, num_shot, num_labelperimage, must_exist_labels, total_label_episode):
    total_label = total_label_episode
    labels = []
    labelmap = {}
    lbl_count = 0
    label_to_idx = {}

    label_count = {}
    query_label = {}

    for jj in range(len(must_exist_labels)):
        query_label[must_exist_labels[jj]] = 1


    for ii in range(len(label_strs)):
        #
        candidates = [int(b) for b in label_strs[ii].split(',')]
        priority_candidates = []
        else_candidates = []
        for candidate in candidates:
            if query_label.get(candidate) != None:
                priority_candidates.append(candidate)
            else :
                else_candidates.append(candidate)
            # if labelmap.get(candidate) == None:
            #     labelmap[candidate] = lbl_count
            #     lbl_count += 1
            if label_count.get(candidate) == None:
                label_count[candidate] = 0
            else:
                label_count[candidate] += 1

        priority_candidates = np.array(priority_candidates)
        labels.append(priority_candidates)
 


    random_label = torch.randperm(total_label) + 1 # 1 base
    relabel = []#labels.clone()
    all_labels = labels#.view(-1)


    #RELABEL and LABEL to IDXS mapping
    for jj in range(len(all_labels)):
        new_label = []
        cc = 0
        for ii in all_labels[jj]:
            if cc > 2:
                break
            if labelmap.get(ii) == None:
                labelmap[ii] = lbl_count
                lbl_count += 1 
            new_label.append(random_label[labelmap[ii]]) #back to 0..N base
            cc += 1

        new_labels = np.array(new_label)[:num_labelperimage]
        if jj > num_query-1: #indexing is just for support set
            for new_label in new_labels:
                nlbl_idxzero = new_label - 1
                if label_to_idx.get(nlbl_idxzero) == None:
                    label_to_idx[nlbl_idxzero] = []
                label_to_idx[nlbl_idxzero].append(jj) #with query and support set
        relabel.append(new_labels)

    padded_labels = pad_labels(relabel, maks=3)

    new_label_onehot = one_hot(torch.from_numpy(padded_labels).long(), total_label+1)

    new_label_onehot_cutzero = new_label_onehot[:, :, 1:]

    return new_label_onehot_cutzero[:num_query], new_label_onehot_cutzero[num_query:], label_to_idx


def create_mask(idxs, label_size):
    if idxs.shape[0] > 0:
        one_hot_labels = torch.zeros_like(idxs)
        temp_lbl = idxs.unsqueeze(-1)
        one_hot_labels = one_hot_labels.unsqueeze(-1).repeat(1, label_size)
        one_hot_labels.scatter_(1, temp_lbl.data, 1)
        one_hot_labels = 1.0 - one_hot_labels

        final_mask = one_hot_labels[0].clone()
        for onehot in one_hot_labels:
            final_mask*=onehot
    else:
        final_mask = torch.ones(label_size)

    return final_mask



def process_label_coco_fewshot_definedlabel(label_strs, num_query, must_exist_labels, total_label_episode):
    total_label = total_label_episode
    labels = []
    labelmap = {}
    lbl_count = 0
    label_to_idx = {}


    query_label = {}

    for jj in range(len(must_exist_labels)):
        #if must_exist_labels[jj] > 0:
        query_label[must_exist_labels[jj]] = 1

    #get related labels as must_exist_label described
    for ii in range(len(label_strs)):
        candidates = [int(b) for b in label_strs[ii].split(',')]
        related_labels  = []
        for candidate in candidates:
            if query_label.get(candidate) != None:
                related_labels.append(candidate)


        related_labels = np.array(related_labels)
        labels.append(related_labels)

    random_label = torch.randperm(total_label) + 1 # 1 base
    relabel = []#labels.clone()
    all_labels = labels#.view(-1)


    #RELABEL and LABEL to IDXS mapping
    for jj in range(len(all_labels)):
        new_label = []
        for ii in all_labels[jj]:
            if labelmap.get(ii) == None:
                labelmap[ii] = lbl_count
                lbl_count += 1
            new_label.append(random_label[labelmap[ii]]) #back to 0..N base


        new_labels = np.array(new_label)
        if jj > num_query-1: #indexing is just for support set
            for new_label in new_labels:
                nlbl_idxzero = new_label - 1
                if label_to_idx.get(nlbl_idxzero) == None:
                    label_to_idx[nlbl_idxzero] = []
                label_to_idx[nlbl_idxzero].append(jj) #with query and support set
        relabel.append(new_labels)

    padded_labels = pad_labels(relabel, maks=10)

    new_label_onehot = one_hot(torch.from_numpy(padded_labels).long(), total_label+1)

    new_label_onehot_cutzero = new_label_onehot[:, :, 1:]

    return new_label_onehot_cutzero[:num_query], new_label_onehot_cutzero[num_query:], label_to_idx



def acc_topk_definedlabel(logits, label,k=3):
    label_size = logits.shape[-1]
    sample_size = logits.shape[0]
    _, pred = torch.topk(logits, k=k, dim=-1)
    if torch.cuda.is_available():

        #label_onehot = one_hot(label.view(-1, 3), label_size).sum(1)
        pred_onehot = one_hot(pred, label_size).sum(1)
        #
        gt_size = label.sum().float()
        all = torch.sum(pred_onehot*label).float()/(gt_size)
        #return (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return all
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()




def accuracy_calc(output, target, label_size, greaterthan=0.1, topk=10):
 
    o_p= o_r= of1 = 0
    num_query = target.shape[0]
    for jj in range(num_query):
        topk_size = topk#label_size//2
        predtopk = 3

 
        prediction    = torch.topk(output[jj], topk_size, dim=-1)#torch.topk(F.softmax(output, dim=1), 10, dim=1)
        filter        = prediction[0].eq(greaterthan) + prediction[0].gt(greaterthan)
        prediction_index         = torch.mul(prediction[1]+1, filter.type(torch.cuda.LongTensor))
        
        extend_eye_mat           = torch.cat((torch.zeros(1, label_size), torch.eye(label_size)), 0)
        prediction_label         = extend_eye_mat[prediction_index.view(-1)].view(-1, topk_size, label_size).sum(dim=1)
 
        correct_prediction_label = (target[jj].cpu().byte() & prediction_label.byte()).type(torch.FloatTensor)

        #count the sum of label vector
        sum_prediction_label         = prediction_label.sum()#.sum(dim=0)
        sum_correct_prediction_label = correct_prediction_label.sum()#.sum(dim=0)
        sum_ground_truth_label       = target[jj].cpu().sum()#.sum(dim=0)

        o_p_temp = 0
        if sum_prediction_label.sum() > 0.0:
            o_p_temp = torch.div(sum_correct_prediction_label.sum(), sum_prediction_label.sum())
        o_r_temp = torch.div(sum_correct_prediction_label.sum(), sum_ground_truth_label.sum())
        of1_temp = 0.0
        if o_p_temp + o_r_temp > 0:
            of1_temp = torch.div(2 * o_p_temp * o_r_temp, o_p_temp + o_r_temp)

        o_p += o_p_temp
        o_r += o_r_temp
        of1 += of1_temp
  
    o_p = o_p/num_query
    o_r = o_r/num_query
    of1 = 0.0
    if o_p + o_r > 0.0:
        of1 = torch.div(2 * o_p * o_r, o_p + o_r)

    return o_p, o_r, of1#, c_p, c_r, cf1



def accuracy_calc_dpp(output, target, label_size, greaterthan=0.1):

    topk_size = 10#label_size//2

    if output.shape[0] < 1:
        return 0.0, 0.0, 0.0
    elif output.shape[0] < topk_size:
        topk_size = output.shape[0]
    prediction    = torch.topk(output, topk_size, dim=-1)#torch.topk(F.softmax(output, dim=1), 10, dim=1)
    filter        = prediction[0].eq(greaterthan) + prediction[0].gt(greaterthan)
    prediction_index         = torch.mul(prediction[1]+1, filter.type(torch.cuda.LongTensor))
    extend_eye_mat           = torch.cat((torch.zeros(1, label_size), torch.eye(label_size)), 0)
    prediction_label         = extend_eye_mat[prediction_index.view(-1)].view(-1, topk_size, label_size).sum(dim=1)

    correct_prediction_label = (target.cpu().byte() & prediction_label.byte()).type(torch.FloatTensor)
  
    #count the sum of label vector
    sum_prediction_label         = prediction_label.sum()
    sum_correct_prediction_label = correct_prediction_label.sum()#.sum(dim=0)
    sum_ground_truth_label       = target.cpu().sum()#.sum(dim=0)



    o_p = 0
    if sum_prediction_label > 0.0:
        o_p = torch.div(sum_correct_prediction_label.sum(), sum_prediction_label)
    o_r = torch.div(sum_correct_prediction_label.sum(), sum_ground_truth_label.sum())
    of1 = 0.0
    if o_p + o_r > 0:
        of1 = torch.div(2 * o_p * o_r, o_p + o_r)

    return o_p, o_r, of1#, c_p, c_r, cf1




def calc_rankloss(scores, gt, margin=1.):

    batch_size = gt.shape[0]
    # Y = []
    # nonY = []
    loss = []
    for i in range(batch_size):
        gt_idxs = torch.nonzero(gt[i]).squeeze()
        non_idxs = (gt[i] == 0).nonzero().squeeze()
        Y = scores[i, gt_idxs]
        nonY = scores[i, non_idxs]
        nony_sum = nonY.shape[0]
        y_sum = Y.shape[0]

        Y = Y.unsqueeze(0).repeat(nony_sum, 1).view(-1)
        nonY = nonY.unsqueeze(-1).repeat(1, y_sum).view(-1)
        #lossf = torch.mean((margin + nonY - Y).clamp(0))
        lossf = torch.log(1 + torch.mean(torch.exp(margin + nonY - Y)))
        loss.append(lossf)

    loss = sum(loss)/batch_size


    return loss




def triplet_loss(scores, W, gt, margin=2.0):
    batch_size = gt.shape[0]
    # Y = []
    # nonY = []
    loss = []
    for i in range(batch_size):
        gt_idxs = torch.nonzero(gt[i]).squeeze()

        Y = scores[i, gt_idxs]
        selected_w = W[gt_idxs, :][:, gt_idxs]
        positiveloss = (Y.unsqueeze(1).repeat(1, Y.shape[0]) - Y.unsqueeze(0).repeat(Y.shape[0], 1))**2
        positiveloss = torch.sum(selected_w*positiveloss)


        loss.append(positiveloss)
 

    loss = 0.5*sum(loss)/batch_size


    return loss


def similarity_loss(scores, W, gt, margin=2.0):
    batch_size = gt.shape[0]
    # Y = []
    # nonY = []
    loss = []
    for i in range(batch_size):
        gt_idxs = torch.nonzero(gt[i]).squeeze()

        Y = scores[i, gt_idxs]
        selected_w = W[gt_idxs, :][:, gt_idxs]
        positiveloss = (Y.unsqueeze(1).repeat(1, Y.shape[0]) - Y.unsqueeze(0).repeat(Y.shape[0], 1))**2
        positiveloss = torch.sum(selected_w*positiveloss)

        loss.append(positiveloss)
 
    loss = 0.5*sum(loss)/batch_size


    return loss





def accuracy_calc_AND(output, target, label_size, greaterthan=0.1, topk=10):

    o_p= o_r= of1 = 0
    num_query = target.shape[0]
    for jj in range(num_query):
        topk_size = topk#label_size//2
        predtopk = 3
        if output[jj].shape[0] < 1:
            return 0.0, 0.0, 0.0
        elif output[jj].shape[0] < topk_size:
            topk_size = output.shape[0]
        prediction    = torch.topk(output[jj], topk_size, dim=-1)#torch.topk(F.softmax(output, dim=1), 10, dim=1)
        filter        = prediction[0].eq(greaterthan) + prediction[0].gt(greaterthan)
        prediction_index         = torch.mul(prediction[1]+1, filter.type(torch.cuda.LongTensor))

        extend_eye_mat           = torch.cat((torch.zeros(1, label_size), torch.eye(label_size)), 0)
        prediction_label         = extend_eye_mat[prediction_index.view(-1)].view(-1, topk_size, label_size).sum(dim=1).cuda()

        prediction_label = prediction_label*2 - 1 #to give it minus 1
        target_new = target[jj]*2 - 1 #to give it minus 1

        correct_prediction_label = (prediction_label*target_new).type(torch.FloatTensor)
        of1 = of1 + torch.mean(correct_prediction_label)
 

    return o_p/num_query, o_r/num_query, of1/num_query