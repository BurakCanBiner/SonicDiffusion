import torch
import torch.nn as nn
import torch.nn.functional as F
# from ldm.modules.attention import SpatialTransformer, BasicTransformerBlock


class Adapter(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], grid_dims = [64,32,16,8], adapter_input_dim = 1024, device="cuda", out_resolutions = [4096, 1024, 256, 64]):
        super(Adapter, self).__init__()
        self.channels = channels
#         self.adapter
        self.input_adapters = []
        self.middle_adapters = []
        self.output_adapters = []
        
        self.inner_channel = 77*8
        self.inner_res = 64 
        self.inner_dim = 64

        ################################  7 - 11 ########################################

        ch = channels[3]
        grid_dim = grid_dims[3]
        layer_adapter = nn.Sequential(
            nn.Conv1d(1, self.inner_dim, 17, stride=16, bias=True, padding=2),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(self.inner_dim, self.inner_dim, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.output_adapters.append(layer_adapter)

        ch = channels[1]
        grid_dim = grid_dims[1]
        layer_adapter = nn.Sequential(
            nn.ConvTranspose2d(self.inner_dim, ch // 4, 8, stride=4, padding=2),
            nn.GELU(),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.output_adapters.append(layer_adapter)
        
        # level 8 -> 640, 64, 64
        ch = channels[1]
        grid_dim = grid_dims[1]
        layer_adapter = nn.Sequential(
            nn.ConvTranspose2d(self.inner_dim, ch // 4, 16, stride=8, padding=4),
            nn.GELU(),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.output_adapters.append(layer_adapter)

        # for 3 blocks -> 320, 64, 64 , levels 9, 10, 11
        for j in range(3):
            ch = channels[0]
            grid_dim = grid_dims[0]
            layer_adapter = nn.Sequential(        
                nn.ConvTranspose2d(self.inner_dim, ch // 4, 16, stride=8, padding=4),
                nn.GELU(),
                nn.Conv2d(ch // 4, ch, 3, 1, 1)
            )
            layer_adapter.to(device)
            self.output_adapters.append(layer_adapter) 
        
        self.model = nn.ModuleList(self.input_adapters + self.middle_adapters + self.output_adapters)

    def forward(self, x):
        output_features = []

        x_init = self.output_adapters[0](x)   
        for module in self.output_adapters[1:]:
            adapter_out = module(x_init)
            output_features.append(adapter_out)
            
        return None, None, output_features
    
            