import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import torch
from PIL import ImageEnhance
from dataloader.additional_transforms import ImageJitter

THIS_PATH = osp.dirname(__file__)
#ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
PATH = '/home/csimon/research/data/imaterialist/'
# IMAGE_PATH = osp.join(PATH, '')
# SPLIT_PATH = osp.join(PATH, '')

transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)


class IMaterialistSet(Dataset):
    """ Usage:
    """
    def __init__(self, path, setname, args, manualSeed=1, aug=False):


        self.aug = aug
        #if manualSeed is None:
        manualSeed = 8601#random.randint(1, 10000)
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        #torch.manual_seed(opt.manualSeed)
    #if opt.cuda:
        torch.cuda.manual_seed_all(manualSeed)

        PATH = path
        IMAGE_PATH = osp.join(PATH, 'images')
        #IMAGE_PATH = osp.join(PATH, 'val2017') # for train eval
        SPLIT_PATH = osp.join(PATH, 'split')
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1
        label_str = []

        self.ids = []

        for l in lines:
            #bb =  l.split('|')
            image_id, lbls = l.split('|')
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
            self.image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args['modeltype'] == 'ResNet':
            self.image_size = 224 #224
            self.transform = transforms.Compose([
                transforms.Resize(224), #224
                #transforms.RandomCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')


        #self.image_size = image_size
        self.normalize_param = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225])
        self.jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)

        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        self.transform_aug = transforms.Compose(transform_funcs)


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
        img_temp = Image.open(path).convert('RGB')

        if self.aug == True:
            image = self.transform_aug(img_temp)
        else:
            image = self.transform(img_temp)
        return (image, label_i)


    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


    def jitter(self, img):
        transforms = [(transformtypedict[k], self.jitter_param[k]) for k in self.jitter_param]
        out = img
        randtensor = torch.rand(len(transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out

    def parse_transform(self,  transform_type):
        if transform_type=='ImageJitter':
            method = ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size)
        elif transform_type=='CenterCrop':
            return method(self.image_size)
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()