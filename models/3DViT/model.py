import torch
import torch.nn as nn
from data.pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction, farthest_point_sample, index_points, square_distance
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, to_2tuple
from timm.models.registry import register_model
from timm.models.vision_transformer import VisionTransformer, _cfg
from torch import nn
from einops import rearrange, repeat
from functools import partial
from models.DeIT import *

class SACBlock(nn.Module):
  def __init__(self, inplanes, expand1x1_planes=None, bn_d = 0.1):

    super(SACBlock, self).__init__()
    self.inplanes = inplanes
    self.bn_d = bn_d

    self.attention_x = nn.Sequential(
            nn.Conv2d(3, 9 * self.inplanes, kernel_size = 7, padding = 3),
            nn.BatchNorm2d(9 * self.inplanes, momentum = 0.1),
            )

    self.position_mlp_2 = nn.Sequential(
            nn.Conv2d(9 * self.inplanes, self.inplanes, kernel_size = 1),
            nn.BatchNorm2d(self.inplanes, momentum = 0.1),
            nn.ReLU(inplace = True),
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(self.inplanes, momentum = 0.1),
            nn.ReLU(inplace = True),
            )

  def forward(self, input1, input2, input3):
    xyz = input1 ### coordinate map
    new_xyz= input2  ### coordinate map
    feature = input3  ### coordinate map with r and intensity
    N,C,H,W = feature.size()

    new_feature = F.unfold(feature, kernel_size = 3, padding = 1).view(N, -1, H, W)
    attention = F.sigmoid(self.attention_x(new_xyz))
    new_feature = new_feature * attention
    new_feature = self.position_mlp_2(new_feature)
    fuse_feature = new_feature + feature
   
    return xyz, new_xyz, fuse_feature

def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 
    
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]

    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)

    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points



class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2
        

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class PointEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.num_class
        d_points = cfg.input_dim
        self.conv1 = nn.Conv1d(d_points, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=cfg.embed_dim // 4)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=cfg.embed_dim // 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = x.permute(0, 2, 1)
        new_xyz_0, new_feature = sample_and_group(npoint=1024, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        #new_xyz_1, new_feature = sample_and_group(npoint=1024, nsample=32, xyz=new_xyz_0, points=feature) 
        #feature_1 = self.gather_local_1(new_feature)
        return new_xyz_0, feature_0, new_xyz_0, feature_0

class AMSoftmaxLayer(nn.Module):
    """AMSoftmaxLayer"""
    def __init__(self,
                 in_feats,
                 n_classes,
                 s=30.):
        super(AMSoftmaxLayer, self).__init__()
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)
    def forward(self, x):
        B,N,C = x.shape
        x = x.view(-1,C)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm) * self.s
        return costh.view(B,N,-1)

class PointTransformerCls(VisionTransformer):
    '''
        3D Feature
    '''
    __valid_model = {
            'deit_tiny_patch16_224': {
                'patch_size':16,
                'embed_dim':192,
                'depth':12,
                'num_heads':3,
                'mlp_ratio':4,
                'qkv_bias':True,
                'norm_layer':partial(nn.LayerNorm, eps=1e-6)
            },
            'deit_small_patch16_224': {
                'patch_size':16,
                'embed_dim':384,
                'depth':12,
                'num_heads':6,
                'mlp_ratio':4,
                'qkv_bias':True,
                'norm_layer':partial(nn.LayerNorm, eps=1e-6)
            },
            'deit_base_patch16_224': {
                'patch_size':16,
                'embed_dim':768,
                'depth':12,
                'num_heads':3,
                'mlp_ratio':4,
                'qkv_bias':True,
                'norm_layer':partial(nn.LayerNorm, eps=1e-6)
            },
            'vit_base_patch16_224_21k': {
                'patch_size':16,
                'embed_dim':768,
                'depth':12,
                'num_heads':3,
                'mlp_ratio':4,
                'qkv_bias':True,
                'norm_layer':partial(nn.LayerNorm, eps=1e-6)
            },
            'vit_large_patch16_224': {
                'patch_size':16,
                'embed_dim':768,
                'depth':12,
                'num_heads':3,
                'mlp_ratio':4,
                'qkv_bias':True,
                'norm_layer':partial(nn.LayerNorm, eps=1e-6)
            },
            }
    __valid_model_pretrain_dict_url = {
        'deit_tiny_patch16_224': "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        'deit_small_patch16_224': "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        'deit_base_patch16_224': "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        'vit_base_patch16_224_21k': "./3rd_party/ViT-PyTorch/jax_to_pytorch/weights/B_16.pth"
    }

    def __init__(self, cfg):
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.transformer_backbone = cfg.model.transformer_backbone
        self.pretrained = cfg.model.pretrained
        transformer_dict = self.__load_transformer_config()
        super().__init__(
            patch_size= transformer_dict['patch_size'],
            embed_dim= transformer_dict['embed_dim'],
            depth= transformer_dict['depth'],
            num_heads= transformer_dict['num_heads'],
            mlp_ratio= transformer_dict['mlp_ratio'],
            qkv_bias= transformer_dict['qkv_bias'],
            norm_layer= transformer_dict['norm_layer'],
        )
        self.default_cfg = _cfg()
        self.url = self.__valid_model_pretrain_dict_url[self.transformer_backbone]
        self.dist_token = None

        self.n_classes = n_c
        cfg.embed_dim = transformer_dict['embed_dim']
        # load weight
        print(self.transformer_backbone)
        self.__load_backbone_weight()

        # change patch embedding layer
        self.patch_embed = PointEmbed(cfg)

        # replace last head layer
        if cfg.model.head =='AMSoftmax':
            self.head = AMSoftmaxLayer(self.embed_dim // 4, self.n_classes)
        else:
            self.head = nn.Linear(self.embed_dim // 4, self.n_classes)

        # Setup different positional embedding
        # TODO: Add correct positional embedding
        self.pos_embed_type = 'default'

        self.transition_downs = nn.ModuleList()
        for i in range(2):
            channel = self.embed_dim // 4 * 2 ** (i+1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** i, nneighbor, [channel // 2 + 3, channel, channel]))
        
        self.transition_ups = nn.ModuleList()
        for i in reversed(range(2)):
            channel = self.embed_dim // 4 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
        
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, self.embed_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 4, self.embed_dim // 4)
        )

        self.fc_pos_embed = nn.Sequential(
            nn.Linear(3, self.embed_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 4, self.embed_dim // 4)
        )
        

    def __load_transformer_config(self):

        if self.transformer_backbone not in self.__valid_model:
            raise ValueError("Unknown transformer backbone name!")

        return self.__valid_model[self.transformer_backbone]


    def __load_backbone_weight(self):
        if self.pretrained:
            print(self.url)
            if '21k' in self.transformer_backbone:
                pretrained_dict = torch.load(self.url, map_location="cpu")
                #pretrained_dict = fit_dict(pretrained_dict)
            else:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url=self.url,
                    map_location="cpu", check_hash=True
                    )
                pretrained_dict = checkpoint["model"]
            # load partial dict_file (except for pos_embed and last layer)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # key_list=  list(pretrained_dict.keys())
            # for k in key_list:
            #     if 'blocks.0.attn' in k:
            #         del pretrained_dict[k]

            #print(pretrained_dict.keys())
            #model_dict.update(pretrained_dict)

            self.load_state_dict(pretrained_dict)



    def forward_features(self, x):
        pos_embedding = self.pos_embed_type
        if pos_embedding is None or pos_embedding=="default" or pos_embedding=="no_embed":
            #_,_,xyz, f = self.patch_embed(x)
            #print(f.shape)
            #f=f.transpose(1,2)
            xyz, f= x[...,:3], self.fc1(x)
            f = self.pos_drop(f + self.fc_pos_embed(xyz))

            xyz_0, points_0 = self.transition_downs[0](xyz, f)
            xyz_1, points_1 = self.transition_downs[1](xyz_0, points_0)
            x = points_1

            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            if self.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

            # Difference starting from here:

            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            x = x[:, 1:]
            x = self.transition_ups[0](xyz_1, x, xyz_0, points_0)
             # print(xyz_0.shape, x.shape, xyz.shape, f.shape)
            x = self.transition_ups[1](xyz_0, x, xyz, f)
            return x.mean(1)
        else:
            raise ValueError("Unknown positional embedding scheme!")

    def forward(self,x):
        """
        Input b * n_points * (3 + n_object_class)
        Output b * n_points * n_part_class
        """
        
        x = self.forward_features(x)
        x = self.head(x)
        return x



class PointTransformerSeg(VisionTransformer):
    '''
        3D Feature
    '''
    __valid_model = {
            'deit_tiny_patch16_224': {
                'patch_size':16,
                'embed_dim':192,
                'depth':12,
                'num_heads':3,
                'mlp_ratio':4,
                'qkv_bias':True,
                'norm_layer':partial(nn.LayerNorm, eps=1e-6)
            },
            'deit_small_patch16_224': {
                'patch_size':16,
                'embed_dim':384,
                'depth':12,
                'num_heads':6,
                'mlp_ratio':4,
                'qkv_bias':True,
                'norm_layer':partial(nn.LayerNorm, eps=1e-6)
            },
            'deit_base_patch16_224': {
                'patch_size':16,
                'embed_dim':768,
                'depth':12,
                'num_heads':3,
                'mlp_ratio':4,
                'qkv_bias':True,
                'norm_layer':partial(nn.LayerNorm, eps=1e-6)
            },
            'vit_base_patch16_224_21k': {
                'patch_size':16,
                'embed_dim':768,
                'depth':12,
                'num_heads':3,
                'mlp_ratio':4,
                'qkv_bias':True,
                'norm_layer':partial(nn.LayerNorm, eps=1e-6)
            },
            'vit_large_patch16_224': {
                'patch_size':16,
                'embed_dim':768,
                'depth':12,
                'num_heads':3,
                'mlp_ratio':4,
                'qkv_bias':True,
                'norm_layer':partial(nn.LayerNorm, eps=1e-6)
            },
            }
    __valid_model_pretrain_dict_url = {
        'deit_tiny_patch16_224': "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        'deit_small_patch16_224': "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        'deit_base_patch16_224': "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        'vit_base_patch16_224_21k': "./3rd_party/ViT-PyTorch/jax_to_pytorch/weights/B_16.pth"
    }

    def __init__(self, cfg):
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.transformer_backbone = cfg.model.transformer_backbone
        self.pretrained = cfg.model.pretrained
        transformer_dict = self.__load_transformer_config()
        super().__init__(
            patch_size= transformer_dict['patch_size'],
            embed_dim= transformer_dict['embed_dim'],
            depth= transformer_dict['depth'],
            num_heads= transformer_dict['num_heads'],
            mlp_ratio= transformer_dict['mlp_ratio'],
            qkv_bias= transformer_dict['qkv_bias'],
            norm_layer= transformer_dict['norm_layer'],
        )
        self.default_cfg = _cfg()
        self.url = self.__valid_model_pretrain_dict_url[self.transformer_backbone]
        self.dist_token = None

        self.n_classes = n_c
        cfg.embed_dim = transformer_dict['embed_dim']
        # load weight
        print(self.transformer_backbone)
        self.__load_backbone_weight()

        # change patch embedding layer
        self.patch_embed = PointEmbed(cfg)

        # replace last head layer
        if cfg.model.head =='AMSoftmax':
            print("head is AMsoftmax  #####################")
            self.head = AMSoftmaxLayer(self.embed_dim // 4, self.n_classes)
        else:
            print("head is Linear #####################")
            self.head = nn.Linear(self.embed_dim // 4, self.n_classes)

        # Setup different positional embedding
        # TODO: Add correct positional embedding
        self.pos_embed_type = 'default'

        self.transition_downs = nn.ModuleList()
        for i in range(2):
            channel = self.embed_dim // 4 * 2 ** (i+1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** i, nneighbor, [channel // 2 + 3, channel, channel]))
        
        self.transition_ups = nn.ModuleList()
        for i in reversed(range(2)):
            channel = self.embed_dim // 4 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
        
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, self.embed_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 4, self.embed_dim // 4)
        )

        self.fc_pos_embed = nn.Sequential(
            nn.Linear(3, self.embed_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 4, self.embed_dim // 4)
        )
        
        self.sac_conv1 = nn.Sequential(
            nn.Conv2d(5, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.sac1 = SACBlock(32)
        self.sac_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.sac2 = SACBlock(64)
        # self.sac_post = nn.Sequential(

        # )
        # self.sac_head = nn.Sequential(
          
        # )

    def __load_transformer_config(self):

        if self.transformer_backbone not in self.__valid_model:
            raise ValueError("Unknown transformer backbone name!")

        return self.__valid_model[self.transformer_backbone]


    def __load_backbone_weight(self):
        if self.pretrained:
            print(self.url)
            if '21k' in self.transformer_backbone:
                pretrained_dict = torch.load(self.url, map_location="cpu")
                #pretrained_dict = fit_dict(pretrained_dict)
            else:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url=self.url,
                    map_location="cpu", check_hash=True
                    )
                pretrained_dict = checkpoint["model"]
            # load partial dict_file (except for pos_embed and last layer)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # key_list=  list(pretrained_dict.keys())
            # for k in key_list:
            #     if 'blocks.0.attn' in k:
            #         del pretrained_dict[k]

            #print(pretrained_dict.keys())
            #model_dict.update(pretrained_dict)

            self.load_state_dict(pretrained_dict)



    def forward_features(self, x):
        pos_embedding = self.pos_embed_type
        if pos_embedding is None or pos_embedding=="default" or pos_embedding=="no_embed":
            # _,_,xyz, f = self.patch_embed(x)
            # print(f.shape)
            # f=f.transpose(1,2)
            xyz, f= x[...,:3], self.fc1(x)
            # print(f'xyz, f shape: {xyz.shape}, {f.shape}')
            f = self.pos_drop(f + self.fc_pos_embed(xyz))
            # print(f'f shape after: {f.shape}')

            xyz_0, points_0 = self.transition_downs[0](xyz, f)
            xyz_1, points_1 = self.transition_downs[1](xyz_0, points_0)
            x = points_1
            # print(f'xyz_0, points_0 shape: {xyz_0.shape}, {points_0.shape}')
            # print(f'xyz_1, points_1 shape: {xyz_1.shape}, {points_1.shape}')
            # print(f'x shape: {x.shape}')

            #x = torch.cat((x, sac_feats), dim=1)    #### our addition
    
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            if self.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            # print(f'x shape after concat: {x.shape}')
            # Difference starting from here:

            for blk in self.blocks:
                x = blk(x)
            # print(f'x after transformer shape: {x.shape}')
            x = self.norm(x)
            x = x[:, 1:]
            x = self.transition_ups[0](xyz_1, x, xyz_0, points_0)
            # print(f'x after t up1 shape: {x.shape}')
            # print(xyz_0.shape, x.shape, xyz.shape, f.shape)
            x = self.transition_ups[1](xyz_0, x, xyz, f)
            # print(f'x after t up2 shape: {x.shape}')
            #print(x.shape)
            return x
        else:
            raise ValueError("Unknown positional embedding scheme!")

    def forward(self,x):
        """
        Input b * n_points * (3 + n_object_class)
        Output b * n_points * n_part_class
        """
        # lidar_image  = 
        # lidar_feats1 = self.sac_conv1(lidar_image)
        # sac_feats1 = self.sac1(lidar_image[..., :3], lidar_image[..., :3], lidar_feats1)
        # lidar_feats2 = self.sac_conv2(sac_feats1)
        # sac_feats2 = self.sac2(lidar_image[..., :3], lidar_image[..., :3], lidar_feats2)
        # print(f'lidar_image, lidar_feats1, lidar_feats2: {lidar_image.shape}, {lidar_feats1.shape}, {lidar_feats2.shape}')
        # print(f'sac_feats1, sac_feats2: {sac_feats1.shape}, {sac_feats2.shape}')
        # print(x.shape)
        x = self.forward_features(x)
        # sac_feats = self.sac_post(sac_feats2)
        # x = self.forward_features(x, sac_feats)
        # sac_head_feats = self.sac_head(sac_feats2)
        # y = torch.cat((x, sac_head_feats), dim=-1)
        # print(f'y shape: {y.shape}')
        x = self.head(x)
        # print(f'final shape of x: {x.shape}')
        return x

