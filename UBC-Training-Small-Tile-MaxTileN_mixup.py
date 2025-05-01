# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# +
# # !pip install --upgrade timm
# # !pip install --upgrade segmentation_models_pytorch

# +
# Change image size limit before import opencv
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

import warnings
warnings.filterwarnings("ignore")

import pandas as pd 
import numpy as np
import torch
from torch_geometric.utils.scatter import scatter 
import cv2
import timm
from PIL import Image
Image.MAX_IMAGE_PIXELS = pow(2,40)

from matplotlib import pyplot as plt 
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from pandarallel import pandarallel
  # to encourage unequal number of tiles
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchmetrics import Accuracy
import datetime
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.strategies import FSDPStrategy
import gc
from torch.distributed.fsdp import MixedPrecision
from transformers import (get_linear_schedule_with_warmup, 
                          get_cosine_schedule_with_warmup, 
                          get_cosine_with_hard_restarts_schedule_with_warmup,
                          get_constant_schedule_with_warmup)
# +

pandarallel.initialize(use_memory_fs=False, nb_workers=os.cpu_count(), progress_bar=True)

# set([x.split('_')[0].split('.')[0] for x in timm.list_models('*', pretrained=True)])

#timm.list_models('*resnet*', pretrained=True)   
# -

# !nvidia-smi 

# +
args = dict(
    seed=42, 
    model='vit_large_patch14_clip_336',   
    lr=2e-5, #5e-5,       
    weight_decay=1e-2,                     
    log_dir='../weights/logs',
    num_workers=6,#min(16, os.cpu_count()),   
    epochs=10,       
    batch_size=4, #32,                
    val_batch_size=8,                       
    accumulate_grad_batches=1,              
    gpus='0',     
    patience=100, #5,       
    precision='16-mixed',     
    scheduler='onecycle', #'plateau', #'onecycle',               
    n_splits=5,  
    gem_p=3.0, #1.0,  
    dropout=0.5, #0.0,                   
    repeats=2, #1,      
    num_tiles=12,           
    num_tiles_val=48,                         
    tile_size=336, #224|256|384|512                    
    tile_overlap=True, #False,  
    add_other_cls=False, #True, 
    new_augment=False, 
    data_path='../data/ubc-small/small',  # train_pruned4/ | small/                
    use_std=False, 
    loss='multilabel', #multilabel|multiclass   
    dice_weight=1.0, #0.2, #1.0,
    focal_weight=1.0, #0.2, #1.0,    
    entropy_weight=1.0, #0.6, #1.0,     
    warmup_steps=100,   
    run_folds=[3], 
)

args = argparse.Namespace(**args)              

# +
# BASE_PATH = 'images/'
# BASE_PATH = 'small/'
 
BASE_PATH = args.data_path

# +
df_train = pd.read_csv('../data/train_folds.csv')#[:10].reset_index(drop=True)          
 
df_train['img_path'] = df_train['image_id'].apply(lambda image_id: f"{BASE_PATH}/{image_id}.png")
# -

df_train.shape

df_train.head()

df_train.dtypes

df_train['label'].value_counts() #normalize=True) ** -1

# +
# pd.__version__

# +
cls_df = (df_train['label'].value_counts(normalize=True) ** -1).to_frame().reset_index()

if args.add_other_cls:
    cls_df = pd.concat(
        [cls_df, pd.DataFrame([['Other', cls_df.proportion.max()*1.5]], columns=['label', 'proportion'])],
        axis=0,
        ignore_index=True,
    )

cls_df['proportion'] /= cls_df['proportion'].min()
cls_df
# - 

cls_map = cls_df.reset_index().set_index('label')[['index']].to_dict()['index']
cls_map

cls_map_inv = cls_df[['label']].to_dict()['label']
cls_map_inv

cls_weights = torch.tensor(cls_df['proportion'].values).float()
cls_weights

# +
# cls_map = {key: index for index, key in enumerate(['CC', 'EC', 'HGSC', 'LGSC', 'MC', 'Other'])}  

# cls_map_inv = {index: key for index, key in enumerate(['CC', 'EC', 'HGSC', 'LGSC', 'MC', 'Other'])}  
# -

args.num_classes  = len(cls_weights)

#df_train = pd.read_csv('../data/combined_gdc_folds.csv')#[:10].reset_index(drop=True)                 
 
#df_train['img_path'] = df_train['image_id'].apply(lambda image_id: f"{BASE_PATH}/{image_id}.png")   

#pd.read_csv('../data/sample_submission.csv')

# +
import numpy as np
import cv2
import gc
from einops import repeat
import gc


def get_mask(image):
    # using BGR2GRAY on RGB work best for the tma
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = rgb2gray(image)
    
    # get corner values for segmentation
    h, w = gray.shape
    h1 = h//100
    w1 = w//100
    corner_values = np.mean([gray[:h1, :w1], gray[-h1:, :w1], gray[:h1, -w1:], gray[-h1:, -w1:]], axis=0).mean()
    
    # get the mask
    if corner_values > 128:
        corner_values = 255 - corner_values
        gray = 255 - gray

    mask = np.abs(gray-corner_values) > corner_values
    return mask.astype('uint8')


def get_patch_bbox(mask, CROP_SIZE=224, OVERLAP=64):
    H, W = mask.shape
    bboxes = []
    probs = []
    for y in range(0, H-CROP_SIZE+1, CROP_SIZE-OVERLAP):
        for x in range(0, W-CROP_SIZE+1, CROP_SIZE-OVERLAP):
            bbox = [x, y, x+CROP_SIZE, y+CROP_SIZE]
            prob = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].mean()
            if prob != 0:
                bboxes.append(bbox)
                probs.append(prob)
    return np.array(bboxes), np.array(probs)


def get_patch_bbox(img_path, CROP_SIZE=224, OVERLAP=64): #, include_std=False):
    img = np.array(Image.open(img_path))
    mask = get_mask(img)
    
    H, W = mask.shape
    bboxes = []
    probs = []
    stds = []
    for y in range(0, H-CROP_SIZE+1, CROP_SIZE-OVERLAP):
        for x in range(0, W-CROP_SIZE+1, CROP_SIZE-OVERLAP):
            bbox = [x, y, x+CROP_SIZE, y+CROP_SIZE]
            prob = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].mean()
            if  prob > 0.0: #0.5:
#                 if include_std:
#                 x = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#                 m = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#                 std_value = np.mean([x[...,t][m].std() for t in range(3)])
                
#                 x = np.ma.array(
#                     img[bbox[1]:bbox[3], bbox[0]:bbox[2]], 
#                     mask=repeat((mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]==0), 'h w -> h w c', c=3),
#                 )
#                 std_value = x.std(axis=(0,1)).mean()
                
                img_ = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                mask_ = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                std_value = img_[mask_.astype('bool')].std(axis=0).mean()
                
                stds.append(std_value)
#                     del x
#                     gc.collect()
                bboxes.append(bbox)
                probs.append(prob)
                
#     if include_std:
    return np.array(bboxes), np.array(probs), np.array(stds)
#     return np.array(bboxes), np.array(probs)


def get_all_patches(img_path, overlap=args.tile_overlap): #, include_std=args.include_std): 
#     mask = get_mask(np.array(Image.open(img_path)))
    result = dict()
    CROP_SIZE = args.tile_size
    for CROP_SIZE in [args.tile_size]: #[224, 256, 384, 512]:
        OVERLAP = CROP_SIZE//4 if overlap else 0
#         bboxes, probs = get_patch_bbox(mask, CROP_SIZE=CROP_SIZE, OVERLAP=OVERLAP)
        out = get_patch_bbox(img_path, CROP_SIZE=CROP_SIZE, OVERLAP=OVERLAP) #, include_std=include_std)
        bboxes = out[0]
        probs = out[1]
#         if include_std:
        stds = out[2]
#         scores = probs*stds
#         sort_order1 = (scores).argsort()[::-1]
#         else:
#         sort_order = probs.argsort()[::-1]
#         bboxes = bboxes[sort_order]
#         probs = probs[sort_order]
        result.update({
            f'bboxes{CROP_SIZE}': bboxes,
            f'probs{CROP_SIZE}': probs,
            f'stds{CROP_SIZE}': stds,
#             f'scores{CROP_SIZE}': scores,
        })
    return result

from functools import reduce as f_reduce

def factors(n):    
    return set(f_reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def get_values(n):
    f = np.sort(list(factors(n)))
    n = len(f)
    n1 = n//2 if n%2==0 else (n+1)//2
    f = np.array([f[:(n+1)//2], f[n//2:][::-1]])
    loc = f[:, f.sum(axis=0).argmin()].tolist()
    return loc



class UBCDataset(Dataset):
    def __init__(self, df, transforms=None, repeats=1, tile_size=224, is_train=False, num_tiles=16, use_std=False):
        #assert tile_size in [224, 256, 384, 512] 
        self.ids = df['image_id'].tolist()
        self.img_paths = df['img_path'].tolist()
        self.bboxes = df[f'bboxes{tile_size}']
        self.probs = df[f'probs{tile_size}']
        self.stds = df[f'stds{tile_size}']
        self.use_std = use_std
            
        self.is_train = is_train
        self.num_tiles = num_tiles
        self.transforms = transforms
        self.repeats = repeats
        if 'label' in df.columns:
            self.labels = df['label'].tolist()
        
        self.mixup_indexes = df.loc[df['label'].isin(['CC','LGSC','EC','MC'])].index
        print(self.mixup_indexes)
        
    def __len__(self):
        return len(self.ids) * self.repeats
    
    def __getitem__(self, real_idx):
        real_idx = real_idx % len(self.ids)
        random_idx = self.mixup_indexes[random.randrange(len(self.mixup_indexes))] 
        _id = self.ids[real_idx]
        
        idxs = [random_idx, real_idx] if self.is_train and random.random() <= .15 else [real_idx]       
        
        image_stacks = [] 
        label_stacks = [] 
                
        for idx in idxs:
            
           
            img_path = self.img_paths[idx]
            img = cv2.imread(img_path,cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #[min_row:max_row, min_col:max_col] 
            
            scores = self.probs[idx]
            if self.use_std:
    #             scores = self.probs[idx] * self.stds[idx]
                scores = self.probs[idx] + (self.stds[idx]/100).clip(0,1)
            
            if self.is_train:
                mask = (self.probs[idx] > 0)
                valid_k = torch.arange(len(self.bboxes[idx]))
                valid_k = valid_k[mask]
                n_valid = len(valid_k)
                n = np.random.choice(
                    valid_k, 
                    size=min(n_valid, self.num_tiles),
                    replace=False, 
                    p=scores[valid_k]/scores[valid_k].sum(), 
                )
            else:
    #             sort_order = scores.argsort()[::-1]
    #             bboxes = self.bboxes[sort_order]
    #             stds = self.stds[sort_order] bnb
                
    #             n_valid = len(self.probs[idx][self.probs[idx] > 0])
    #             n = np.tile(np.arange(n_valid), self.num_tiles//n_valid + 1)[:self.num_tiles]
    #             n = np.tile(np.arange(n_valid), (3*self.num_tiles)//n_valid + 1)[:3*self.num_tiles]
                
                sort_order = scores.argsort()[::-1]
                mask = (self.probs[idx] > 0)[sort_order]
                valid_sort = sort_order[mask]
                n_valid = len(valid_sort)
    #             n = np.tile(valid_sort, self.num_tiles//n_valid + 1)[:self.num_tiles]
                n = valid_sort[:min(n_valid, self.num_tiles)]
            
            bboxes = self.bboxes[idx][n]
    #         probs = self.probs[idx][n]
            if img.shape[-1] != 3:
                img = torch.stack([img[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] for bbox in bboxes])
            else:
                img = np.stack([img[bbox[1]:bbox[3], bbox[0]:bbox[2]] for bbox in bboxes])
            
            if self.transforms is not None:
                img = torch.stack([self.transforms(image=imgk)['image'] for imgk in img])
            
            image_stacks.append(img)
            label_stacks.append(F.one_hot(torch.tensor(cls_map[self.labels[idx]]).long(), len(cls_map)).float()) 
        
        if len(image_stacks) > 1:
            min_length = min([img.shape[0] for img in image_stacks])   
            #print('min_length -> ', min_length)  
            
            image_stacks = torch.stack([img[:min_length,:,:,:].unsqueeze(0) for img in image_stacks])
            image_stacks = image_stacks.mean(0)
            #print(image_stacks.shape) 
            label_stacks = torch.stack([label.unsqueeze(0) for label in label_stacks]).mean(0) 
            #print(label_stacks)
            
            img = image_stacks.squeeze(0)
            label = label_stacks
        else:
            img = image_stacks[0] 
            label = label_stacks[0].unsqueeze(0)
            
        sample = dict(
            image_id=_id,
            image=img,
            #probs=self.probs[idx][n],
            #stds=self.stds[idx][n],
        )
        
        #if hasattr(self, 'labels'):
        sample['train_label'] = label
        sample['label'] = cls_map[self.labels[idx]]
            
        return sample


# -

def get_transforms(train=False, debug=False):
    augmentations = []
    if train:
        augmentations.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
#             A.RandomShadow (p=0.5),
#             A.RandomSunFlare(p=0.5),
#             A.GaussianBlur(blur_limit=(3, 7), sigma_limit=[0.1, 3], p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.3),
#             A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.5, -0.1), rotate_limit=0, border_mode=0, p=0.5),  # added Mar 2   
            A.ColorJitter(p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.25),
        ])
        
        if args.new_augment:
            aug8p3 = A.OneOf([
                    A.Sharpen(p=0.3),
                    A.ToGray(p=0.3),
                    A.CLAHE(p=0.3),
                ], p=0.5)

            augmentations.extend([
                    A.ShiftScaleRotate(rotate_limit=15, scale_limit=0.1, border_mode=cv2.BORDER_REFLECT, p=0.5),
                    #A.Resize(image_size, image_size),
                    aug8p3,
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                ])
    
    if not debug:
        augmentations.append(
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225), 
                max_pixel_value=255.0,
            ),  # using basic imagenet statistics (default)
        )
    augmentations.extend([
        ToTensorV2(),
    ])
    
    transforms = A.Compose(
        augmentations,
    )
    
    return transforms


def plot_sample(dst, idx=None):
    if idx is None:
        idx = np.random.randint(len(dst), size=(1,))[0]
    
    sample = dst[idx]
    images = sample['image']
    if images.shape[1] == 3:
        images = images.permute(0,2,3,1)
    
    r, c = get_values(images.shape[0]) # get factors
    
    _, axes = plt.subplots(r, c, figsize=(c*5, r*5))
    axes = axes.flatten()
    
    title = str(sample['image_id'])
    if 'label' in sample.keys():
        title += f': {cls_map_inv[sample["label"]]}'
        
    for k in range(images.shape[0]):
        image = images[k]
        if images.shape[0] == 3:
            image = image.permute(1,2,0)
        axes[k].imshow(image)
#         prob = sample['probs'][k]
#         axes[k].set_title(f"{title}")  # : {prob: 0.4f}")
        axes[k].set_title(f"{title} : p={sample['probs'][k]: 0.4f} : s={sample['stds'][k]: 0.4f}")




# define a custom collate function for efficient merging of samples
def collate_fn(original_batch):
#     ['image_id', 'image', 'probs', 'stds', 'label']
    image_id = [sample['image_id'] for sample in original_batch]
    image = torch.cat([sample['image'] for sample in original_batch], dim=0) 
    # create a tensor for saving the samples identifier
    index = torch.cat([torch.ones(sample['image'].shape[0])*k for k, sample in enumerate(original_batch)], dim=-1)
    batch = dict(
        image_id=image_id,
        image=image,
        index=index.long(),
    )
    batch['train_label'] = torch.cat([sample['train_label'] for sample in original_batch], dim=0) 
    #print('batch[\'label\'] -> ',batch['label'].shape)   
    if 'label' in original_batch[0].keys():
        label = torch.tensor([sample['label'] for sample in original_batch]).float()
        batch['label'] = label
    return batch


# +
from functools import wraps
import gc

def flush_and_gc(f):
    @wraps(f)
    def g(*args, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()
        return f(*args, **kwargs)
    return g


# +
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gem(x, p=1, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=1, eps=1e-6, flatten=True):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            return ret.flatten(1)
        return ret

    def __repr__(self):
        return (
                self.__class__.__name__
                + "("
                + "p="
                + "{:.4f}".format(self.p.data.tolist()[0])
                + ", "
                + "eps="
                + str(self.eps)
                + ")"
        )


# +
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
class Multisample_Dropout(nn.Module):
    def __init__(self):
        super(Multisample_Dropout, self).__init__()
        self.dropout = nn.Dropout(.1) 
        self.dropouts = nn.ModuleList([nn.Dropout((i+1)*.1) for i in range(5)])
        
    def forward(self, x, module):
        x = self.dropout(x)
        return torch.mean(torch.stack([module(dropout(x)) for dropout in self.dropouts],dim=0),dim=0) 
class Bi_RNN(nn.Module):
    def __init__(self, size, hidden_size, layers=1):
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(size, hidden_size, num_layers=layers, bidirectional=True, bias=True, batch_first=True) 
                                
    def forward(self, x):
        x, hidden = self.rnn(x)
        return torch.cat((x[:,-1,:self.hidden_size], x[:,0,self.hidden_size:]), dim=-1) 
    

class TileModel(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=6, dropout=0.0, gem_p=1.0):        
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='') 
        #self.backbone.set_grad_checkpointing() 

        #self.pool = GeM(p=gem_p)
        
        self.dropout = nn.Dropout(p=dropout)
        self.num_features = self.backbone.num_features
        self.fc = nn.Linear(self.backbone.num_features, num_classes)
        init_layer(self.fc)
        
    def forward(self, x): 
        
        x = self.backbone(x)
        #x = self.dropout(x.mean(1), self.fc)
        x = self.dropout(x.mean(1))
        #x = self.dropout(x.mean(1))
        x = self.fc(x) 
        
        return x

# +
#soft summing module for tiles aggregation
from torch_geometric.utils.scatter import scatter  # to encourage unequal number of tiles


def soft_sum(x, dim, index, a=1.0):
    # find the parametric softmax indexing/probablity
    y = torch.exp(a*x)
    ym = scatter(y, index=index, dim=dim, reduce='sum')[index]
    prob = y / ym
    # multiply by prob and then sum
    x1 = scatter(x * prob, index=index, dim=0, reduce='sum')
    return x1


class SoftSum(nn.Module):
    def __init__(self, a=1.0):
        super().__init__()
        self.a = Parameter(torch.ones(1) * a)
        
    def forward(self, x, index, dim=0):
        # find the parametric softmax indexing/probablity
        y = torch.exp(self.a*x)
        ym = scatter(y, index=index, dim=dim, reduce='sum')[index]
        prob = y / ym
        # multiply by prob and then sum
        x1 = scatter(x * prob, index=index, dim=0, reduce='sum')
        return x1


# +
class MultiClassCustomLoss(nn.Module):
    def __init__(self, 
                num_classes=args.num_classes, 
                dice_weight=args.dice_weight, 
                focal_weight=args.focal_weight, 
                entropy_weight=args.entropy_weight
                ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.entropy_weight = entropy_weight
        
        self.dice = smp.losses.DiceLoss(mode='multiclass', classes=num_classes, smooth=0.0)
        self.focal = smp.losses.FocalLoss(mode='multiclass', alpha=None, gamma=2.0, normalized=True)
        self.entropy = nn.CrossEntropyLoss(label_smoothing=0.0, weight=cls_weights) 
    
    def forward(self, pred, label):
        #label = label.long()
        #print('pred.shape) -> ',pred.shape)
        #print('label.shape) -> ',label.shape)
        dice = self.dice(pred, label)
        focal = self.focal(pred, label)
        entropy = self.entropy(pred, label)
        
#         loss = dice + focal + entropy
        loss = (dice * self.dice_weight) + (focal * self.focal_weight) + (entropy * self.entropy_weight)
        
        return loss


class MultiLabelCustomLoss(nn.Module):
    def __init__(self, 
                num_classes=args.num_classes, 
                dice_weight=args.dice_weight, 
                focal_weight=args.focal_weight, 
                entropy_weight=args.entropy_weight
                ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.entropy_weight = entropy_weight
        
        self.dice = smp.losses.DiceLoss(mode='multilabel', smooth=0.1)
        self.focal = smp.losses.FocalLoss(mode='multilabel', alpha=None, gamma=2.0, normalized=True)
        self.entropy = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1, pos_weight=cls_weights)  
    
    def forward(self, pred, label):
        #label = label.long()
        #print('pred.shape) -> ',pred.shape)
        #print('label.shape) -> ',label.shape)
        
        #if label.shape != pred.shape:
        #    label = F.one_hot(label.long(), num_classes=self.num_classes) # to one_hot
        
        dice = self.dice(pred, label)
        focal = self.focal(pred, label)
        entropy = self.entropy(pred, label)
        
#         loss = dice + focal + entropy
        loss = (dice * self.dice_weight) + (focal * self.focal_weight) + (entropy * self.entropy_weight) 
        
        return loss


# +
# loss = MultiLabelCustomLoss()
# pred = torch.rand(32, 5)
# label = torch.randint(2, size=(32,))
# loss(pred, label)

# +
from einops import rearrange, reduce
    
    
class UBCModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        
        num_classes = args.num_classes
        self.net = TileModel(args.model, num_classes, args.dropout, args.gem_p)   
        self.aggregate = SoftSum()
        self.criterion = {
            'multilabel': MultiLabelCustomLoss(),
            'multiclass': MultiClassCustomLoss(),
        }[args.loss]
#         self.accuracy = {
#             'multilabel': Accuracy(task='multilabel', num_labels=num_classes, average='macro', ignore_index=0),  #balanced,
#             'multiclass': Accuracy(task='multiclass', num_classes=num_classes, average='macro'),  #balanced
#         }[args.loss]
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='macro')  #balanced
        
        self.labels_val = []
        self.preds_val = []
        
        self.preds_train = []
        self.labels_train = []
        
    def forward(self, image, index):
        x = self.net(image)
        out = self.aggregate(x, index)
#         b = image.shape[0]
#         if image.ndim == 5:
#             image = rearrange(image, 'b n c h w -> (b n) c h w')
#         out = self.net(image)
#         out = reduce(out, '(b n) c -> b c', reduction='mean', b=b)
        return out #self.net(image)
    
#     @flush_and_gc
    def training_step(self, batch, batch_idx=0):
        image = batch['image']
        index = batch['index']
        label = batch['label']
        train_label = batch['train_label']
        
        pred = self(image, index)
        
        loss = self.criterion(pred, train_label) 
        #acc = self.accuracy(pred, label)
        
        self.preds_train.append(pred)
        self.labels_train.append(label)
        
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True, sync_dist=True)
        self.log('loss', loss, batch_size=len(label), prog_bar=True, on_step=True, sync_dist=True)
        #self.log('acc', acc, batch_size=len(label), prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def on_train_epoch_end(self):
        
        preds = torch.cat(self.preds_train, 0)
        labels = torch.cat(self.labels_train, 0)
        
        acc = self.accuracy(preds, labels)
        
        self.log('acc', acc, batch_size=len(labels), prog_bar=True, on_epoch=True, sync_dist=True) 
        
        self.preds_train = []
        self.labels_train = []
    
#     @flush_and_gc
    def validation_step(self, batch, batch_idx=0):
        image = batch['image']
        index = batch['index'] 
        label = batch['label']
        train_label = batch['train_label']

        pred = self(image, index)
        
        loss = self.criterion(pred, train_label) 
        
        self.preds_val.append(pred) 
        self.labels_val.append(label)
        
        self.log('val_loss', loss, batch_size=len(label), prog_bar=True, on_epoch=True, sync_dist=True)
        
    def on_validation_epoch_end(self):
        
        preds = torch.cat(self.preds_val, 0)
        labels = torch.cat(self.labels_val, 0)
        
        acc = self.accuracy(preds, labels)
        
        self.log('val_acc', acc, batch_size=len(labels), prog_bar=True, on_epoch=True, sync_dist=True)
        
        self.preds_val = []
        self.labels_val = []
        
    def configure_optimizers(self):
        #optimizer = optim.AdamW(self.parameters(), lr=self.hparams.args.lr, weight_decay=self.hparams.args.weight_decay)
        optimizer = optim.AdamW(get_parameters(self, self.hparams.args))
        #scheduler_name = self.hparams.args.scheduler

        total_steps=self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.args.warmup_steps, num_training_steps=total_steps)
        interval = 'step'

#         scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-7)
        
        scheduler = {
            'scheduler': scheduler,
            'interval': interval,
            'monitor': 'val_acc'
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def predict_step(self, batch, batch_idx=0):
        index = batch['index']
        image = batch['image']
        
        image_id = batch['image_id']
        
        pred = self(image, index) 
        if self.hparams.args.loss == 'multiclass':
            pred = pred.softmax(dim=1)
        else:
            pred = pred.sigmoid()
        return {'image_id': image_id, 'pred': pred.cpu().numpy()}


# -

"""df_train['fold'] = -1
skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
for fold, (t_idx, v_idx) in enumerate(skf.split(range(len(df_train)), y=df_train[['label', 'is_tma']].astype('str').sum(axis=1))):
    df_train.loc[v_idx, 'fold'] = fold"""

df_train.fold.value_counts()

df_train.groupby(['fold', 'is_tma', 'label']).size()


# +
# df = df_train.query('fold!=0').reset_index(drop=True)
# label = df[['is_tma', 'label']].astype('str').sum(axis=1)
# min_label_count = label[label.str.startswith('False')].value_counts().min()
# tma_classes = label[label.str.startswith('True')].unique()
# df_list = [df]
# for tma_class in tma_classes:
#     df1 = df[label == tma_class]
#     df1 = df1.sample(n=min_label_count-len(df1), replace=True, random_state=42)
#     df_list.append(df1)
    
# df = pd.concat(df_list, axis=0, ignore_index=True)

# +
# df

# +
# tma_classes

# +
# label[label.str.startswith('False')].value_counts()

# +
# min_label_count

# +
# label.value_counts()

# +
# df.groupby(['fold', 'is_tma', 'label']).size()
# -

def oversample(df):
    label = df[['is_tma', 'label']].astype('str').sum(axis=1)
    min_label_count = label[label.str.startswith('False')].value_counts().min()
    tma_classes = label[label.str.startswith('True')].unique()
    df_list = [df]
    for tma_class in tma_classes:
        df1 = df[label == tma_class]
        df1 = df1.sample(n=min_label_count-len(df1), replace=True, random_state=42)
        df_list.append(df1)

    df = pd.concat(df_list, axis=0, ignore_index=True)
    return df


# +
import gc
import inspect
import shutil
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar


def remove_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass


def free_memory(to_delete: list):
    calling_namespace = inspect.currentframe().f_back

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()   
        torch.cuda.empty_cache()


def get_callbacks(args, fold=None):
    start_name = "" 
    if fold is not None:
        start_name = f"fold{fold}-"
        
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=start_name + "{epoch}-{val_loss:0.4f}-{val_acc:0.4f}", 
        monitor='val_acc',
        verbose=False,
        save_last=False,
        save_top_k=1, 
        mode='max',
        save_weights_only=True
    )
    
    early_stop_callback = pl.callbacks.EarlyStopping( 
        monitor="val_acc",
        patience=args.patience,
        verbose=True,
        mode='max',
        strict=True,
        check_finite=True,
        check_on_train_epoch_end=False
    )
    prog_rate = TQDMProgressBar(refresh_rate=1)  
    
    return [
        checkpoint_callback,
        early_stop_callback,
        prog_rate,
    ]


# -

try:
    torch.set_float32_matmul_precision('high')
except:
    pass

# +
import random


def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    pl.seed_everything(seed, workers=True)


# -

# initial reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2" 
# pl.seed_everything(args.seed, workers=True)
fix_seed(args.seed)


# +
# samples = next(iter(val_loader))

# +
# samples['index'].shape

# +
def get_parameters(model, args):

    parameter_settings = [] 
    
    parameter_settings.extend(get_parameter_section([(n, p) for n, p in model.named_parameters()], lr=args.lr, wd=args.weight_decay))    

    return parameter_settings

def get_parameter_section(parameters, lr, wd):      
    
    parameter_settings = []
    unfreeze_layers = ['aggregate.a','net.fc','.attn.qkv.','net.pool']                
    
    for no, (n,p) in enumerate(parameters):  
                
        if any([(var in n) for var in unfreeze_layers]):       
            p.requires_grad = True    
        else:
            p.requires_grad = False      
    
            
        if no <= 100:   
            this_lr = 1e-5     
        elif no <= 148:  
            this_lr = 1e-5    
        elif no <= 244:  
            this_lr = 1e-4        
        elif no <= 294:   
            this_lr = 1e-4                                    
        elif no <= 305: 
            this_lr = 1e-4                    
                        
        #this_lr = 3e-5                         
        this_wd = 0.0 if 'bias' in n else wd   
  
        parameter_setting = {"params" : p, "lr" : this_lr, "weight_decay" : this_wd}        

        parameter_settings.append(parameter_setting)
        
        print(f'no {no} | params {n} | lr {this_lr} | weight_decay {this_wd} | requires_grad {p.requires_grad}')      

    return parameter_settings

# +

if __name__ == "__main__":
    
    results = df_train.img_path.parallel_apply(get_all_patches)
    df_train = pd.concat([df_train, pd.DataFrame.from_records(results)], axis=1)
    count = df_train[df_train.columns[df_train.columns.str.startswith('bboxes')]].applymap(len)  
    count.columns = [column.replace('bboxes', 'count') for column in count]
    df_train = pd.concat([df_train, count], axis=1)
    #df_train
    # -
        
    date_time = datetime.datetime.now().strftime("%m%d-%H%M")

    name = args.model
    version = name + '_' + date_time

    # SET LOGGER
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        name=name,
        version=version, 
    )

    print('\n')
    print(args)
    print(version)
    print('\n')

    checkpoint_paths = []
    fold_scores = []
    oof_dfs = []

    #for fold in range(args.n_splits):
    for fold in args.run_folds:
        print(f"\n#####Starting fold {fold}#####\n")
        
        # make reproducible
    #     pl.seed_everything(args.seed, workers=True)
        fix_seed(args.seed)
        
        # call the model here
        model = UBCModel(args)
        
        # dataloaders
        dst_train= UBCDataset(
    #         df=df_train[df_train['fold']!=fold].reset_index(drop=True),
            df=oversample(df_train[df_train['fold']!=fold].reset_index(drop=True)),
            transforms=get_transforms(train=True),
            is_train=True,
            repeats=args.repeats,
            num_tiles=args.num_tiles, tile_size=args.tile_size,
            use_std=args.use_std,
        )
        dst_val= UBCDataset(
            df=df_train[df_train['fold']==fold].reset_index(drop=True), 
            transforms=get_transforms(train=False),
            is_train=False,
            repeats=1,
            num_tiles=args.num_tiles_val, #args.num_tiles,  
            tile_size=args.tile_size,
            use_std=args.use_std,
        )
        
        train_loader = DataLoader(
            dst_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            dst_val, batch_size=args.val_batch_size, num_workers=args.num_workers, shuffle=False,
            collate_fn=collate_fn,
        )
        
        # Initialize simple Callbacks
        callbacks = get_callbacks(args, fold=fold)

        # Initialize a trainer
        trainer = pl.Trainer(
            callbacks=callbacks,
            accelerator='gpu',
            #strategy='ddp_find_unused_parameters_true',  
            #strategy=FSDPStrategy(mixed_precision=MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float16, buffer_dtype=torch.float16, keep_low_precision_grads=True, cast_forward_inputs=True, cast_root_forward_inputs=True), sharding_strategy='FULL_SHARD'),  
            devices=[int(t) for t in args.gpus.split(',')],
            max_epochs=args.epochs, 
            logger=tb_logger,  
            num_sanity_val_steps=0,
            accumulate_grad_batches=args.accumulate_grad_batches,  
            precision=args.precision,
            gradient_clip_val=1.0,  #1.0 if args.optimizer != 'sam' else None,    
            gradient_clip_algorithm='norm',  
        )


        # Train the model
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader) 

        # include the paths in the list
        checkpoint_path = trainer.checkpoint_callback.best_model_path
        score = trainer.early_stopping_callback.best_score
        
        checkpoint_paths.append(checkpoint_path)
        fold_scores.append(score)
        
        # OOF preds
        preds = trainer.predict(model, dataloaders=val_loader)
        oof_df = pd.DataFrame(preds).explode(column=['image_id', 'pred'], ignore_index=True)
        oof_df['fold'] = fold
        oof_dfs.append(oof_df) 
        
        print(checkpoint_path)
        free_memory([model, trainer])

    oof_df = pd.concat(oof_dfs, axis=0, ignore_index=True)
    oof_df['label'] = oof_df['pred'].apply(lambda pred: pred.argmax())

    oof_path = os.path.join(os.sep.join(checkpoint_path.split(os.sep)[:-2]), f"{version}.csv")   
    oof_df.to_csv(oof_path, index=False)  
    
    # +
    ckpt_paths = []
    for checkpoint_path in checkpoint_paths:
        ckpt_path = checkpoint_path.replace('=', '')     
        os.rename(checkpoint_path, ckpt_path)       
        ckpt_paths.append(ckpt_path)  
        
    checkpoint_paths = ckpt_paths                     




