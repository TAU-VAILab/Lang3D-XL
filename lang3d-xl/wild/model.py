import torch
from torch import nn
import numpy as np
from typing import Iterable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from wild.encoding_tcnn import HashEncoding


EPS = 1e-6
TEMPERATURE = 0.1


def log_clamp(tensor: torch.Tensor, min_val: float = EPS, max_val: float = None):
    return torch.log(torch.clamp(tensor, min=min_val, max=max_val))


class WildModel(nn.Module):
    def __init__(self, images_names: Iterable[str], num_features=512, render_semantic=True) -> None:
        super().__init__()
        self.render_semantic = render_semantic
        self.in_training = True
        hidden_dim = max(51 + num_features, 64)
        
        self.enc = HashEncoding(
            in_dim=3, num_levels=12, log2_hashmap_size=19, min_res=16, max_res=2048)
        
        self.mlp = nn.Sequential(
            nn.Linear(51 + num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4 + num_features))
        
        self.latent_vectors = {}
        for image_name in images_names:
            attr_name = f'latent_{image_name}'
            setattr(self, attr_name, Parameter(torch.randn(24)))
            self.latent_vectors[image_name] = getattr(self, attr_name)
    
    def forward(self, center, color, features, image_name):
        encoded_center = self.enc(center)
        mlp_input = torch.cat((encoded_center,
                               color,
                               features,
                               self.latent_vectors[image_name].unsqueeze(0).repeat((color.shape[0], 1))),
                              dim=-1)
        output = self.mlp(mlp_input)
        new_color = output[:, :3]
        delta_opacity = output[:, 3]
        new_features = output[:, 4:]
        
        u = torch.as_tensor(np.random.choice([0.0, 1.0])) if self.training else torch.tensor(0.5)
        u = u.to(delta_opacity.dtype).to(delta_opacity.device)

        delta_opacity = torch.sigmoid(
            (1 / TEMPERATURE) * (
                log_clamp(torch.abs(delta_opacity) + log_clamp(u) - log_clamp(1-u))
                ))

        return new_color, delta_opacity.unsqueeze(-1), new_features


class WildFeatEncModel(nn.Module):
    def __init__(self, images_names: Iterable[str], num_features=512, num_dino_features=0,
                 render_semantic=True, vucabulary_size=10, is_hash=False, do_attention=False,
                 is_vis_hash=True, max_win_scale_factor=-1.0, mlp_feat_decay=1e-5) -> None:
        super().__init__()
        self.render_semantic = render_semantic
        self.in_training = True
        self.num_features = num_features
        self.num_dino_features = num_dino_features
        self.images_names = images_names
        self.hash_enc_layers = None
        self.is_vis_hash = is_vis_hash
        self.max_win_scale_factor = max_win_scale_factor
        self.mlp_feat_decay = mlp_feat_decay
        
        if self.is_vis_hash:
            self.enc = HashEncoding(
                in_dim=3, num_levels=12, log2_hashmap_size=19, min_res=16, max_res=2048)
            
            hidden_dim = 64  # max(51 + vucabulary_size, 64)
            self.mlp = nn.Sequential(
                nn.Linear(51, hidden_dim), #  + vucabulary_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4))  # + vucabulary_size))
            
            for image_name in self.images_names:
                attr_name = f'latent_{image_name}'
                setattr(self, attr_name, Parameter(torch.randn(24)))
                self.register_parameter(attr_name, getattr(self, attr_name))
            
            self.latent_vectors = LatentVectors(self.images_names)
            
        
        self.is_enc = is_hash
        # if not self.is_enc:
        #     self.feat_vocabulary = Parameter(torch.randn((1, vucabulary_size, num_features)))
        # else:
        if self.is_enc:
            self.feat_enc = HashEncoding(
                in_dim=4, num_levels=16, features_per_level=8,
                log2_hashmap_size=19, min_res=16, max_res=2048, dtype=torch.float)
            self.feat_mlp = nn.Sequential(
                nn.Linear(num_features, num_features),
                nn.ReLU(),
                nn.Linear(num_features, num_features),
                nn.ReLU(),
                nn.Linear(num_features, num_features + num_dino_features))
        
        # self.feat_var = Parameter(torch.ones((1, 3)))
        # self.feat_mean = Parameter(torch.zeros((1, 3)))
        self.do_attention = do_attention
        if do_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.Conv2d(num_features, num_features, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.Conv2d(num_features, 1, kernel_size=3, padding='same'))
    
    def get_latent_vec(self, image_name):
        # attr_name = f'latent_{image_name}'
        # return getattr(self, attr_name)
        return self.latent_vectors.get_latent_vec(image_name)
        
    def forward(self, center: torch.Tensor, color, features, image_name):
        if self.is_vis_hash:
            encoded_center = self.enc(center)
            mlp_input = torch.cat((encoded_center,
                                color,
                                # features,
                                self.get_latent_vec(image_name).unsqueeze(0).repeat((color.shape[0], 1))),
                                dim=-1)
            output = self.mlp(mlp_input)
            new_color = color + output[:, :3]
            delta_opacity = output[:, 3]
            new_features = features  # (center - self.feat_mean) * self.feat_var # center.detach() # output[:, 4:]
            # with torch.no_grad():
            #     new_features = center.detach()
            
            u = torch.as_tensor(np.random.choice([0.0, 1.0])) if self.training else torch.tensor(0.5)
            u = u.to(delta_opacity.dtype).to(delta_opacity.device)

            delta_opacity = torch.sigmoid(
                (1 / TEMPERATURE) * (
                    log_clamp(torch.abs(delta_opacity) + log_clamp(u) - log_clamp(1-u))
                    ))

            return new_color, delta_opacity.unsqueeze(-1), new_features
        else:
            return color, 0, features
    
    def decode_features(self, encoded_features: torch.Tensor):
        if not self.is_enc:
            return encoded_features
            # return torch.mean(encoded_features.unsqueeze(-1)
            #                   * self.feat_vocabulary, dim=1)
        else:
            feat_shape = encoded_features.shape
            if self.hash_enc_layers is None:
                enc_out_dim = self.feat_enc.get_out_dim()
                self.hash_enc_layer_num = int(np.ceil(self.num_features / enc_out_dim))
                self.hash_enc_layers = torch.arange(
                    self.hash_enc_layer_num).to(encoded_features.device) / self.hash_enc_layer_num # taking long

            layers = self.hash_enc_layers.repeat(feat_shape[0])
            encoded_features = encoded_features.repeat_interleave(self.hash_enc_layer_num, dim=0)
            encoded_features = torch.cat((layers.unsqueeze(-1), encoded_features), dim=-1)
            return self.feat_mlp(
                self.feat_enc(encoded_features).reshape((feat_shape[0], -1))[:, :self.num_features]
                )
    
    def parameters(self, recurse: bool = True, mode='') -> Iterable[Parameter]:
        params = []
        if self.is_vis_hash:
            params.append({'params': self.enc.parameters(), "name": "wild_enc"}) 
            params.append({'params': self.mlp.parameters(), "name": "wild_mlp"})  # , 'weight_decay': 1e-5}, 
            params.append({'params': self.latent_vectors.parameters(), "name": "wild_latent"}) #,
            # {'params': self.feat_mean, "name": "wild_feat_mean"},
            # {'params': self.feat_var, "name": "wild_feat_var"}]
        # for image_name in self.images_names:
        #     attr_name = f'latent_{image_name}'
        #     params.append({'params': getattr(self, attr_name), "name": f"wild_{attr_name}"})
        if self.is_enc:
            params.append({'params': self.feat_enc.parameters(), "name": "wild_feat_enc"})#, 'weight_decay': 1e-5})
            params.append({'params': self.feat_mlp.parameters(), "name": "wild_feat_mlp", 'weight_decay': self.mlp_feat_decay})
        if self.do_attention:
            params.append({'params': self.attention.parameters(), "name": "wild_feat_attention", 'weight_decay': 1e-5})
        
        return params
    
    def attention_downsample(self, encoded_features: torch.Tensor, size: tuple, get_attention: bool = False,
                             range_factor=None, dropout=0.0):
        feat_height, feat_width = encoded_features.shape[1:]
        win_height, win_width = np.ceil(feat_height / size[0]), np.ceil(feat_width / size[1])
        win_height, win_width = feat_height // size[0], feat_width // size[1]

        stride = max(win_height, 1)
        #range_factor = 7
        
        win_dim = stride
        if range_factor is not None:
            if self.max_win_scale_factor > 0:
                range_factor = min(range_factor, self.max_win_scale_factor)
            win_dim = round(win_dim * range_factor)
        else:
            range_factor = 1
        win_dim = min(int(win_dim * 2 + 1), int(size[0]))

        new_size = stride * size[0] + win_dim - 1, stride * size[1] + win_dim - 1
        encoded_features = F.interpolate(
            encoded_features.unsqueeze(0), size=new_size,
            mode='bilinear', align_corners=False).squeeze(0)

        attention_weight = self.attention(encoded_features.unsqueeze(0)).squeeze(0)
        max_trick = torch.max(attention_weight)
        attention_weight = torch.exp(attention_weight - max_trick)

        if isinstance(size, dict):
            keys = size.keys()  
            attenuated_encoded_features = {key: self.resize_by_attention_weight(
                encoded_features, size[key], attention_weight, range_factor=range_factor,
                stride=stride, win_dim=win_dim, dropout=dropout) for key in keys}
        else:
            attenuated_encoded_features = self.resize_by_attention_weight(
                encoded_features, size, attention_weight, range_factor=range_factor,
                stride=stride, win_dim=win_dim, dropout=dropout)
        
        if get_attention:
            return attenuated_encoded_features, attention_weight
        return attenuated_encoded_features

    def resize_by_attention_weight(self, encoded_features, size, attention_weight,
                                   range_factor, stride, win_dim, dropout=0.0):
        

        #weight = np.random.choice((1.0, 0.0), size=(1,1,win_height,win_width), p=(0.9, 0.1))
        # weight = np.random.choice((1.0, 0.0), size=(1,1,win_dim,win_dim), p=(0.5, 0.5))
        # while np.sum(weight) == 0:
        #     # weight = np.random.choice((1.0, 0.0), size=(1,1,win_height,win_width), p=(0.9, 0.1))
        #     weight = np.random.choice((1.0, 0.0), size=(1,1,win_dim,win_dim), p=(0.5, 0.5))
        
        #sigma = weight.shape[0] / range_factor if range_factor is not None else weight.shape[0] / 5
        #gaussian = create_2d_gaussian(weight.shape[0], sigma, sigma)
            
        curr_size = stride
        pyramid_sizes = [curr_size]
        max_size = round(stride * range_factor)
        while curr_size < max_size:
            curr_size *= np.sqrt(2)
            rounded_size = round(curr_size)
            pyramid_sizes.append(min(rounded_size, max_size))
        pyramid = create_2d_pyramid(pyramid_sizes, size[0], dropout=dropout)
        # delta = pyramid.shape[0] - weight.shape[0]
        # if delta:
        #     pyramid = pyramid[delta - delta // 2: -delta // 2 + 1]
        # weight = weight * pyramid #* gaussian
        weight = pyramid
        
        weight = torch.from_numpy(weight).to(encoded_features.dtype).to(encoded_features.device)
        while len(weight.shape) < 4:
            weight = weight.unsqueeze(0)
        
        attenuated_encoded_features = downsample_to_size(
            (attention_weight * encoded_features).unsqueeze(1), size, 
            weight, stride=(stride, stride), padding_mode='reflect').squeeze(1)
        norm_factor = downsample_to_size(
            attention_weight.unsqueeze(0), size, weight, stride=(stride, stride)).squeeze(0)
        
        # attenuated_encoded_features = torch.nn.functional.conv2d(  # encoded_features + torch.nn.functional.conv2d(
        #     (attention_weight * encoded_features).unsqueeze(1), weight, padding='same').squeeze(1)
        # norm_factor = torch.nn.functional.conv2d( # 1.0 + torch.nn.functional.conv2d(
        #     attention_weight.unsqueeze(0), weight, padding='same').squeeze(0)
        
        attenuated_encoded_features = attenuated_encoded_features / torch.clamp(norm_factor, min=1e-9)
        # attenuated_encoded_features = F.interpolate(
        #     attenuated_encoded_features.unsqueeze(0), size=size,
        #     mode='nearest').squeeze(0)
            
        return attenuated_encoded_features

def downsample_to_size(input_tensor, target_size, kernel, stride=None, padding_mode="reflect"):
    """
    Downsample a tensor to a specific size using torch.nn.functional.conv2d.
    
    Parameters:
    - input_tensor (torch.Tensor): Input tensor of shape (C, H, W).
    - target_size (tuple): Desired output size (H_out, W_out).
    - kernel (torch.Tensor): Convolutional kernel of shape (out_channels, in_channels, kH, kW).
    
    Returns:
    - torch.Tensor: Downsampled tensor.
    """
    # Unpack input dimensions and target size
    _, _, H_in, W_in = input_tensor.shape
    H_out, W_out = target_size
    _, _, kH, kW = kernel.shape

    if stride is None:
        # Compute the required stride
        stride_h = int(np.ceil(H_in / H_out))  # np.ceil((H_in - kH + 1) / H_out))
        stride_w = int(np.ceil(W_in / W_out))  # np.ceil((W_in - kW + 1) / W_out))
        stride = (stride_h, stride_w)

        # Compute the required padding
        pad_h = max((H_out - 1) * stride_h + kH - H_in, 0)
        pad_w = max((W_out - 1) * stride_w + kW - W_in, 0)
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)  # (left, right, top, bottom)

        # Apply padding
        padded_input = F.pad(input_tensor, padding, mode=padding_mode, value=0)  # Add batch dim and pad
    else:
        padded_input = input_tensor

    # Perform convolution
    try:
        output = torch.nn.functional.conv2d(padded_input, kernel, stride=stride)  # Remove batch dim
    except:
        raise ValueError(f'kernel shape: {kernel.shape}, padded input shape: {padded_input.shape}, stride: {stride}')

    return output

def create_2d_pyramid(sizes, min_size, dropout=0.0):
    
    # Create a grid of (x, y) coordinates
    new_sizes = [size * 2 + 1 for size in sizes]
    max_size = min(int(max(new_sizes)), min_size)
    x = np.linspace(- max_size // 2, max_size // 2, max_size)
    y = np.linspace(- max_size // 2, max_size // 2, max_size)
    x, y = np.meshgrid(x, y)
    
    pyramid = 0 * x
    factor = 1.0
    for size in new_sizes:
        window = np.greater_equal(x, -size // 2) * np.less_equal(x, size // 2) * np.greater_equal(y, -size // 2) * np.less_equal(y, size // 2)
        
        drop = dropout  # * min(1.0, size / max_size)
        dropout_map = np.random.choice((1.0, 0.0), size=window.shape, p=(1 - drop, drop))
        while np.sum(dropout_map) == 0:
            dropout_map = np.random.choice((1.0, 0.0), size=window.shape, p=(1 - drop, drop))
        
        pyramid += dropout_map * window * factor
        factor /= 2 * np.sqrt(2)
    
    return pyramid

def create_2d_gaussian(size=100, sigma_x=1.0, sigma_y=1.0, mean_x=0.0, mean_y=0.0):
    """
    Create a 2D Gaussian distribution.
    
    Parameters:
        size (int): Grid size (number of points along each dimension).
        sigma_x (float): Standard deviation along the x-axis.
        sigma_y (float): Standard deviation along the y-axis.
        mean_x (float): Mean along the x-axis.
        mean_y (float): Mean along the y-axis.
    
    Returns:
        gaussian (numpy.ndarray): Values of the 2D Gaussian on the grid.
    """
    # Create a grid of (x, y) coordinates
    x = np.linspace(- size // 2, size // 2, size)
    y = np.linspace(- size // 2, size // 2, size)
    x, y = np.meshgrid(x, y)
    
    # Calculate the 2D Gaussian
    # gaussian = (1 / (2 * np.pi * sigma_x * sigma_y)) * \
    #     np.exp(-((x - mean_x) ** 2 / (2 * sigma_x ** 2) + (y - mean_y) ** 2 / (2 * sigma_y ** 2)))
    gaussian = \
        np.exp(-((x - mean_x) ** 2 / (2 * sigma_x ** 2) + (y - mean_y) ** 2 / (2 * sigma_y ** 2)))
    
    return gaussian

class LatentVectors(nn.Module):
    def __init__(self, images_names: Iterable[str], default_image = None) -> None:
        super().__init__()
        
        self.images_names = images_names
        self.default_image = default_image

        for image_name in images_names:
            attr_name = f'latent_{image_name}'
            setattr(self, attr_name, Parameter(torch.randn(24)))
    
    def get_latent_vec(self, image_name):
        if image_name not in self.images_names:
            image_name = self.default_image if self.default_image \
                else self.images_names[len(self.images_names)//2]
        attr_name = f'latent_{image_name}'
        return getattr(self, attr_name)
        
    def forward(self, image_name):
        return self.get_latent_vec(image_name)
