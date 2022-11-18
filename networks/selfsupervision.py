import torch
import torch.nn as nn
import numpy as np


class LabelCounter(nn.Module):
    def __init__(self, feature_dim=1600):
        super(LabelCounter, self).__init__()
        self.comparator = nn.Sequential(nn.Linear(3*feature_dim, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 20)
                                        )
        self.direct_counter = nn.Sequential(nn.Linear(1*feature_dim, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 10)
                                        )

    ##### USE CROSS ENTROPY
    def forward(self, input, query, support_count_label):
        global_context = input.mean(0)
        global_context = global_context.repeat(input.size(0), 1)
        vec1 = torch.cat((input, global_context), dim=-1)
        allfeature_size = vec1.size(0)
        query_size = query.size(0)
        vec1 = vec1.unsqueeze(0).repeat(query_size, 1, 1)
        query_repeat = query.unsqueeze(1).repeat(1, allfeature_size, 1)
        all_vec = torch.cat((query_repeat, vec1), dim=-1)

        result = self.comparator(all_vec)
        position_int = torch.argmax(result, dim=-1)

        idx_inference = self.voting(position_int, support_count_label)

        #all_features = torch.cat((query, input), dim=0)
        out_label_num = 0#self.direct_counter(all_features)

        return result, idx_inference, out_label_num


    def voting(self, results, count_label):
        results_int = torch.squeeze(results).cpu().int().detach().numpy()

        data_histo = dict()#torch.zeros(results.size(0)).cuda()
        maks = 0
        idx_arr = []

        for qq in range(results.size(0)):
            idx = 0
            for jj in range(results_int[qq].shape[0]):

                res = int(results_int[qq, jj] - count_label[jj])
                if res <= 0:
                    res = 1
                if data_histo.get(res) is None:
                    data_histo[res] = 1
                else:
                    data_histo[res] += 1
                if data_histo[res] > maks:
                    maks = data_histo[res]
                    idx = res
            #if idx > 10:
            #    idx = 9
            #if idx == 0:
            #    idx = 1
                    #print("idx")

            idx_arr.append(idx)

        idx_arr = np.array(idx_arr)
        idx_arr = torch.from_numpy(idx_arr)
        return idx_arr