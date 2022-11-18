import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import torch

THIS_PATH = osp.dirname(__file__)
#ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
PATH = '/home/csimon/research/data/nus-wide'
# IMAGE_PATH = osp.join(PATH, '')
# SPLIT_PATH = osp.join(PATH, '')




class NusWideSet(Dataset):
    """ Usage:
    """
    def __init__(self, path, setname, args, manualSeed=1):

        if manualSeed is None:
            manualSeed = random.randint(1, 10000)
            print("Random Seed: ", manualSeed)
            random.seed(manualSeed)

        #torch.manual_seed(manualSeed)
        #if opt.cuda:
        torch.cuda.manual_seed_all(manualSeed)

        PATH = path
        IMAGE_PATH = osp.join(PATH, 'Flickr')
        SPLIT_PATH = osp.join(PATH, 'split_k2')
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1
        label_str = []

        self.ids = []

        for l in lines:
            image_id, lbls = l.split(' ')
            name = image_id#+'.jpg'
            path = osp.join(IMAGE_PATH, name)
            if not osp.exists(path):
                continue
            # if wnid not in self.wnids:
            #     self.wnids.append(wnid)
            #     lb += 1
            lb +=1
            lbl_ints = [int(x) for x in lbls.split(',')]
            label_str.append(lbls)
            data.append(path)
            label.append(lbl_ints)
            self.ids.append(lb)

        self.data = data
        self.label = label
        self.label_str = label_str
        self.label_ids = self.assign_label_to_id(data, label)
        #self.num_class = len(set(label))

        # Transformation
        if args['modeltype'] == 'ConvNet':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args['modeltype'] == 'ResNet':
            image_size = 224
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

    def assign_label_to_id(self, data, label):

        out = {}

        for i in range(len(data)):
            label_int = label[i]
            for j in range(len(label_int)):
                num=(label_int[j])
                if out.get(num) == None:
                    out[num] = []
                out[num].append(i)

        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label_i = self.data[i], self.label_str[i]
        try:
            image = self.transform(Image.open(path).convert('RGB'))
        except:
            print(self.data[i])
            return
        return (image, label_i)