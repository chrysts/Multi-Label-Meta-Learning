import torch
import numpy as np


class EpisodeSampler():

    def __init__(self, ids, labels, label_ids, query_num, total_label_size, shot_size, n_batch=32, iter=12902):
        self.iter = iter #- (iter % n_batch)
        self.n_batch = n_batch
        self.ids = np.array(ids) #12902
        self.labels = np.array(labels)
        self.label_ids = label_ids
        self.query_num = query_num
        self.total_label_size = total_label_size
        self.shot_size = shot_size
        self.counter = -1

        self.all_unique_labels = [key for key in self.label_ids.keys()]

        self.must_label = np.zeros((self.iter, self.total_label_size))
        self.total_original_label = np.zeros((self.iter, total_label_size*(self.shot_size) + self.query_num))

    def __len__(self):
        return self.iter

    def get_querylabel(self, i):
        return self.must_label[i]

    def get_total_ori_label(self, i):
        return self.total_original_label[i]

    def __iter__(self):
        for ii in range(self.iter):
            randids = torch.randperm(len(self.ids))

            samples = randids[:self.query_num]
            query = torch.from_numpy(self.ids[samples])

            all_labels_list = self.labels[query] # {1,2}, {7,2,3}, {4,5,6}

            required_labels = set()

            all_labels = np.array([])#all_labels_list[0])
            for unique_labels in all_labels_list:
                unique_labels = np.array(unique_labels)
                all_labels = np.concatenate((all_labels, unique_labels), axis=0)
                label_perm = torch.randperm(len(unique_labels))[:2] #minimum two labels for each query
                required_labels.add(unique_labels[label_perm[0]])
                required_labels.add(unique_labels[label_perm[1]])

            all_label_perm = torch.randperm(all_labels.shape[0])


            for jj in range(len(all_labels)):
                if len(required_labels) >= self.total_label_size:
                    break
                required_labels.add(all_labels[all_label_perm[jj]])

            # Add more label to become N labels
            if len(required_labels) < self.total_label_size:
                num_alllabel = len(self.label_ids)
                for jj in range(num_alllabel):
                    if len(required_labels) >= self.total_label_size:
                        break
                    perm = torch.randperm(num_alllabel)[0].item()
                    required_labels.add(self.all_unique_labels[perm])


            supportsets = []#torch.floatTensor(0)
            filled = 0
            all_taken_labels = []
            # print("JJ : " + str(ii))
            # print(required_labels)
            for req_label in required_labels: #misal yang diambil adalah {5,4,3}
                all_taken_labels.append(req_label)
                ids_withassoc_label = np.array(self.label_ids[req_label]) # ids dengan label 5,4,3
                ids_perm = torch.randperm(len(ids_withassoc_label))[:self.shot_size] # how many shot that you want
                taken_ids = ids_withassoc_label[ids_perm]
                if self.shot_size < 2 :
                    taken_ids = np.array([taken_ids])
                #supportsets.append(torch.from_numpy(taken_ids))
                try:
                    if filled > 0:
                        supportsets = torch.cat((supportsets, torch.from_numpy(taken_ids)), dim=0)
                    else:

                        supportsets = torch.from_numpy(taken_ids)
                        filled = 1
                except:
                    print(taken_ids)
                    print(" ")
                    print(req_label)


            all_taken_labels = np.array(all_taken_labels).reshape(-1)

            self.counter = self.counter + 1

            try:
                self.must_label[self.counter, :all_taken_labels.shape[0]] = all_taken_labels
                #supportsets = torch.stack(supportsets).reshape(-1) #ORDER : 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4

                all_set = torch.cat((query, supportsets), dim=0)
            except:
                print(supportsets)
                continue

    
            labels_all = self.labels[all_set]
            all_label_count = []
            for labels_persample in (labels_all):
                all_label_count.append(len(labels_persample))

            all_label_count = np.array(all_label_count)
            #self.total_original_label[self.counter, :] = all_label_count


       
            yield all_set

        self.counter = -1
        #self.must_label = self.must_label*0
        #supportsets = torch.transpose(supportsets, 0, 1)