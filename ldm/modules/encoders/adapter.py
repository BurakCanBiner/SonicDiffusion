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
        
        
#         for ind, (ch, grid_dim) in enumerate(zip(channels, grid_dims)):
            
            
#             nn.Unflatten(1, torch.Size([2, 5, 5]))
# nn.ConvTranspose1d(1, 320, 4, stride=4, padding=0)





        ############### INPUT BLOCKS ##############################
        ch = channels[0]
        grid_dim = grid_dims[0]
        layer_adapter = nn.Sequential(
            nn.ConvTranspose1d(1, ch // 4  , 4, stride=4, bias=True, padding=0),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.input_adapters.append(layer_adapter)
        
        ch = channels[1]
        grid_dim = grid_dims[1]
        layer_adapter = nn.Sequential(
            nn.Conv1d(1, ch // 4 , 17, stride=1, bias=True, padding=8),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.input_adapters.append(layer_adapter)
        
        ch = channels[2]
        grid_dim = grid_dims[2]
        layer_adapter = nn.Sequential(
            nn.Conv1d(1, ch // 4 , 17, stride=4, bias=True, padding=8),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.input_adapters.append(layer_adapter)
    
        ch = channels[3]
        grid_dim = grid_dims[3]
        layer_adapter = nn.Sequential(
            nn.Conv1d(1, ch // 4, 17, stride=16, bias=True, padding=2),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.input_adapters.append(layer_adapter)
        
        ############### MIDDLE BLOCKS ##############################

        ch = channels[3]
        grid_dim = grid_dims[3]
        layer_adapter = nn.Sequential(
            nn.Conv1d(1, ch // 4, 17, stride=16, bias=True, padding=2),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
            )
        self.inner_channel = 77*8
        self.inner_res = 64 
        self.inner_dim = 64

   ############## OUTPUT BLOCKS ONLY #######################
    ################# 4-11 ###############################


        # ch = channels[3]
        # grid_dim = grid_dims[3]
        # layer_adapter = nn.Sequential(
        #     nn.Conv1d(1, self.inner_dim, 17, stride=16, bias=True, padding=2),
        #     nn.GELU(),
        #     nn.Unflatten(2, (grid_dim, grid_dim)),
        #     nn.Conv2d(self.inner_dim, self.inner_dim, 3, 1, 1)
        # )
        # layer_adapter.to(device)
        # self.output_adapters.append(layer_adapter)

        # # level 4 -> 1280, 16, 16  

        # ch = channels[2]
        # grid_dim = grid_dims[2]
        # layer_adapter = nn.Sequential(
        #     nn.ConvTranspose2d(self.inner_dim, ch // 4, 4, stride=2, padding=1),
        #     nn.GELU(),
        #     nn.Conv2d(ch // 4, ch, 3, 1, 1)
        # )
        # layer_adapter.to(device)
        # self.output_adapters.append(layer_adapter)

        # # level 5 -> 1280, 32, 32 
        # ch = channels[2]
        # grid_dim = grid_dims[2],
        # layer_adapter = nn.Sequential(
        #     nn.ConvTranspose2d(self.inner_dim, ch // 4, 8, stride=4, padding=2),
        #     nn.GELU(),
        #     nn.Conv2d(ch // 4, ch, 3, 1, 1)
        # )
        # layer_adapter.to(device)
        # self.output_adapters.append(layer_adapter)


        # # for 2 blocks -> 640, 32, 32 , levels 6, 7
        # for j in range(2):
        #     ch = channels[1]
        #     grid_dim = grid_dims[1]
        #     layer_adapter = nn.Sequential(
        #         nn.ConvTranspose2d(self.inner_dim, ch // 4, 8, stride=4, padding=2),
        #         nn.GELU(),
        #         nn.Conv2d(ch // 4, ch, 3, 1, 1)
        #     )
        #     layer_adapter.to(device)
        #     self.output_adapters.append(layer_adapter)
        
        # # level 8 -> 640, 64, 64
        # ch = channels[1]
        # grid_dim = grid_dims[1]
        # layer_adapter = nn.Sequential(
        #     nn.ConvTranspose2d(self.inner_dim, ch // 4, 16, stride=8, padding=4),
        #     nn.GELU(),
        #     nn.Conv2d(ch // 4, ch, 3, 1, 1)
        # )
        # layer_adapter.to(device)
        # self.output_adapters.append(layer_adapter)


        # # for 3 blocks -> 320, 64, 64 , levels 9, 10, 11
        # for j in range(3):
        #     ch = channels[0]
        #     grid_dim = grid_dims[0]
        #     layer_adapter = nn.Sequential(        
        #         nn.ConvTranspose2d(self.inner_dim, ch // 4, 16, stride=8, padding=4),
        #         nn.GELU(),
        #         nn.Conv2d(ch // 4, ch, 3, 1, 1)
        #     )
        #     layer_adapter.to(device)
        #     self.output_adapters.append(layer_adapter) 


        ################################  7 - 11 ########################################

        # ch = channels[3]
        # grid_dim = grid_dims[3]
        # layer_adapter = nn.Sequential(
        #     nn.Conv1d(1, self.inner_dim, 17, stride=16, bias=True, padding=2),
        #     nn.GELU(),
        #     nn.Unflatten(2, (grid_dim, grid_dim)),
        #     nn.Conv2d(self.inner_dim, self.inner_dim, 3, 1, 1)
        # )
        # layer_adapter.to(device)
        # self.output_adapters.append(layer_adapter)

        # ch = channels[1]
        # grid_dim = grid_dims[1]
        # layer_adapter = nn.Sequential(
        #     nn.ConvTranspose2d(self.inner_dim, ch // 4, 8, stride=4, padding=2),
        #     nn.GELU(),
        #     nn.Conv2d(ch // 4, ch, 3, 1, 1)
        # )
        # layer_adapter.to(device)
        # self.output_adapters.append(layer_adapter)
        
        # # level 8 -> 640, 64, 64
        # ch = channels[1]
        # grid_dim = grid_dims[1]
        # layer_adapter = nn.Sequential(
        #     nn.ConvTranspose2d(self.inner_dim, ch // 4, 16, stride=8, padding=4),
        #     nn.GELU(),
        #     nn.Conv2d(ch // 4, ch, 3, 1, 1)
        # )
        # layer_adapter.to(device)
        # self.output_adapters.append(layer_adapter)


        # # for 3 blocks -> 320, 64, 64 , levels 9, 10, 11
        # for j in range(3):
        #     ch = channels[0]
        #     grid_dim = grid_dims[0]
        #     layer_adapter = nn.Sequential(        
        #         nn.ConvTranspose2d(self.inner_dim, ch // 4, 16, stride=8, padding=4),
        #         nn.GELU(),
        #         nn.Conv2d(ch // 4, ch, 3, 1, 1)
        #     )
        #     layer_adapter.to(device)
        #     self.output_adapters.append(layer_adapter) 
        
        
        
        #################################
        #################################
#     ########### INPUT BLOCKS #################
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[0], bias=True)
#         )
#         layer_adapter.to(device)
#         self.input_adapters.append(layer_adapter)
        
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[0], bias=True)
#         )
#         layer_adapter.to(device)
#         self.input_adapters.append(layer_adapter)
        
        
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[1], bias=True)
#         )
#         layer_adapter.to(device)
#         self.input_adapters.append(layer_adapter)
        
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[1], bias=True)
#         )
#         layer_adapter.to(device)
#         self.input_adapters.append(layer_adapter)
        
    
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[2], bias=True)
#         )
#         layer_adapter.to(device)
#         self.input_adapters.append(layer_adapter)
        
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[2], bias=True)
#         )
#         layer_adapter.to(device)
#         self.input_adapters.append(layer_adapter)
        

#         ############# MIDDLE BLOCKS ###################
        
<<<<<<< HEAD
        layer_adapter = nn.Sequential(
            nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
            nn.GELU(),
            torch.nn.Linear(self.inner_res, out_resolutions[3], bias=True)
>>>>>>> 31e0cee6b294499a442e056c919926632ffd3099
        )
        layer_adapter.to(device)
        self.middle_adapters.append(layer_adapter)
        
<<<<<<< HEAD
        ############### OUTPUT BLOCKS ##############################
        
        ch = channels[3]
        grid_dim = grid_dims[3]
        layer_adapter = nn.Sequential(
            nn.Conv1d(1, ch // 4, 17, stride=16, bias=True, padding=2),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
=======
        ############# OUTPUT BLOCKS ###################
        
        layer_adapter = nn.Sequential(
            nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
            nn.GELU(),
            torch.nn.Linear(self.inner_res, out_resolutions[0], bias=True)
>>>>>>> 31e0cee6b294499a442e056c919926632ffd3099
        )
        layer_adapter.to(device)
        self.output_adapters.append(layer_adapter)
        
<<<<<<< HEAD
        
                
        ch = channels[2]
        grid_dim = grid_dims[2]
        layer_adapter = nn.Sequential(
            nn.Conv1d(1, ch // 4 , 17, stride=4, bias=True, padding=8),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.output_adapters.append(layer_adapter)
    

        ch = channels[1]
        grid_dim = grid_dims[1]
        layer_adapter = nn.Sequential(
            nn.Conv1d(1, ch // 4 , 17, stride=1, bias=True, padding=8),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
=======
        layer_adapter = nn.Sequential(
            nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
            nn.GELU(),
            torch.nn.Linear(self.inner_res, out_resolutions[0], bias=True)
        )
        layer_adapter.to(device)
        self.output_adapters.append(layer_adapter)
=======
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[3], bias=True)
#         )
#         layer_adapter.to(device)
#         self.middle_adapters.append(layer_adapter)
        
#         ############# OUTPUT BLOCKS ###################
        
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[2], bias=True)
#         )
#         layer_adapter.to(device)
#         self.output_adapters.append(layer_adapter)
        
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[2], bias=True)
#         )
#         layer_adapter.to(device)
#         self.output_adapters.append(layer_adapter)
>>>>>>> 228b50c1afa4e32e77d268d2b93b550456490c33
        
        
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[2], bias=True)
#         )
#         layer_adapter.to(device)
#         self.output_adapters.append(layer_adapter)
        
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[1], bias=True)
#         )
#         layer_adapter.to(device)
#         self.output_adapters.append(layer_adapter)
        
    
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[1], bias=True)
#         )
#         layer_adapter.to(device)
#         self.output_adapters.append(layer_adapter)
        
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[1], bias=True)
#         )
#         layer_adapter.to(device)
#         self.output_adapters.append(layer_adapter)
        
        
<<<<<<< HEAD
        layer_adapter = nn.Sequential(
            nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
            nn.GELU(),
            torch.nn.Linear(self.inner_res, out_resolutions[3], bias=True)
>>>>>>> 31e0cee6b294499a442e056c919926632ffd3099
        )
        layer_adapter.to(device)
        self.output_adapters.append(layer_adapter)
        
<<<<<<< HEAD
        ch = channels[0]
        grid_dim = grid_dims[0]
        layer_adapter = nn.Sequential(
            nn.ConvTranspose1d(1, ch // 4  , 4, stride=4, bias=True, padding=0),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
=======
        layer_adapter = nn.Sequential(
            nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
            nn.GELU(),
            torch.nn.Linear(self.inner_res, out_resolutions[3], bias=True)
>>>>>>> 31e0cee6b294499a442e056c919926632ffd3099
        )
        layer_adapter.to(device)
        self.output_adapters.append(layer_adapter)
=======
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[0], bias=True)
#         )
#         layer_adapter.to(device)
#         self.output_adapters.append(layer_adapter)
        
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[0], bias=True)
#         )
#         layer_adapter.to(device)
#         self.output_adapters.append(layer_adapter)
        
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, self.inner_channel, 32, stride=16, bias=True, padding=8),
#             nn.GELU(),
#             torch.nn.Linear(self.inner_res, out_resolutions[0], bias=True)
#         )
#         layer_adapter.to(device)
#         self.output_adapters.append(layer_adapter)        
>>>>>>> 228b50c1afa4e32e77d268d2b93b550456490c33
        
<<<<<<< HEAD
=======
        ##################################
        ##################################
                
        ##########
#         for ind, (ch, grid_dim) in enumerate(zip(channels, grid_dims)):
            
            
#             nn.Unflatten(1, torch.Size([2, 5, 5]))
# nn.ConvTranspose1d(1, 320, 4, stride=4, padding=0)

#         self.inner_dim = 64
#         # common upsmaple first 
        
#         grid_dim = grid_dims[0]
#         layer_adapter = nn.Sequential(
#             nn.ConvTranspose1d(1, self.inner_dim  , 4, stride=4, bias=True, padding=0),
#             nn.GELU(),
#             nn.Unflatten(2, (grid_dim, grid_dim)),
#             nn.Conv2d(self.inner_dim, self.inner_dim, 3, 1, 1),
#             nn.GELU()
#         )
#         layer_adapter.to(device)
#         self.input_adapters.append(layer_adapter)
        
#         ch = channels[0]
#         grid_dim = grid_dims[0]
#         layer_adapter = nn.Sequential(
#             nn.Conv2d(self.inner_dim, ch // 4, 3, stride=2, padding=1),
#             nn.GELU(),
#             nn.Conv2d(ch // 4, ch, 3, 1, 1)
#         )
#         layer_adapter.to(device)
#         self.input_adapters.append(layer_adapter)

#         prev_ch = ch
#         ch = channels[1]
#         grid_dim = grid_dims[1]
#         layer_adapter = nn.Sequential(
#             nn.Conv2d(prev_ch, ch // 4, 3, stride=2, padding=1),
#             nn.GELU(),
#             nn.Conv2d(ch // 4, ch, 3, 1, 1)
#         )
#         layer_adapter.to(device)
#         self.input_adapters.append(layer_adapter)

        
#         prev_ch = ch
#         ch = channels[2]
#         grid_dim = grid_dims[2]
#         layer_adapter = nn.Sequential(
#             nn.Conv2d(prev_ch, ch // 4, 3, stride=2, padding=1),
#             nn.GELU(),
#             nn.Conv2d(ch // 4, ch, 3, 1, 1)
#         )
#         layer_adapter.to(device)
#         self.input_adapters.append(layer_adapter)
        
        
#         prev_ch = ch
#         ch = channels[3]
#         grid_dim = grid_dims[3]
#         layer_adapter = nn.Sequential(
#             nn.Conv2d(prev_ch, ch // 4, 3, stride=2, padding=1),
#             nn.GELU(),
#             nn.Conv2d(ch // 4, ch, 3, 1, 1)
#         )
#         layer_adapter.to(device)
#         self.input_adapters.append(layer_adapter)
        
        
################ downsample first ######################3


    ### downsample then tranpose ###


 ############## INPUT BLOCKS ##############################

        self.inner_dim = 64
    
        # common downsample first  
        ch = channels[3]
        grid_dim = grid_dims[3]
        layer_adapter = nn.Sequential(
            nn.Conv1d(1, self.inner_dim, 17, stride=16, bias=True, padding=2),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(self.inner_dim, self.inner_dim, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.input_adapters.append(layer_adapter)
    
        ch = channels[0]
        grid_dim = grid_dims[0]
        layer_adapter = nn.Sequential(
            nn.ConvTranspose2d(self.inner_dim, ch // 4, 16, stride=8, padding=4),
            nn.GELU(),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.input_adapters.append(layer_adapter)        
        
        ch = channels[1]
        grid_dim = grid_dims[1]
        layer_adapter = nn.Sequential(
            nn.ConvTranspose2d(self.inner_dim, ch // 4, 8, stride=4, padding=2),
            nn.GELU(),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.input_adapters.append(layer_adapter)
        
        ch = channels[2]
        grid_dim = grid_dims[2]
        layer_adapter = nn.Sequential(
            nn.ConvTranspose2d(self.inner_dim, ch // 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.input_adapters.append(layer_adapter)
        
        ch = channels[3]
        grid_dim = grid_dims[3]
        layer_adapter = nn.Sequential(
            nn.Conv2d(self.inner_dim, ch // 4, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.input_adapters.append(layer_adapter)

        
 ############### MIDDLE BLOCKS ##############################

        # common downsample first  
        ch = channels[3]
        grid_dim = grid_dims[3]
        layer_adapter = nn.Sequential(
            nn.Conv1d(1, self.inner_dim, 17, stride=16, bias=True, padding=2),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(self.inner_dim, self.inner_dim, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.middle_adapters.append(layer_adapter)
    
        # set channel size  
        ch = channels[3]
        grid_dim = grid_dims[3]
        layer_adapter = nn.Sequential(
            nn.Conv2d(self.inner_dim, ch // 4, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.middle_adapters.append(layer_adapter)
        
        
 ############# OUTPUT BLOCKS ##############################

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
 

        ch = channels[3]
        grid_dim = grid_dims[3]
        layer_adapter = nn.Sequential(
            nn.Conv2d(self.inner_dim, ch // 4, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.output_adapters.append(layer_adapter)
        
        ch = channels[2]
        grid_dim = grid_dims[2]
        layer_adapter = nn.Sequential(
            nn.ConvTranspose2d(self.inner_dim, ch // 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
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
        
        ch = channels[0]
        grid_dim = grid_dims[0]
        layer_adapter = nn.Sequential(        
            nn.ConvTranspose2d(self.inner_dim, ch // 4, 16, stride=8, padding=4),
            nn.GELU(),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        layer_adapter.to(device)
        self.output_adapters.append(layer_adapter) 
        
        
### Previous implementation with 1dconv-actiation-2dconv

#         ############### INPUT BLOCKS ##############################
#         ch = channels[0]
#         grid_dim = grid_dims[0]
#         layer_adapter = nn.Sequential(
#             nn.ConvTranspose1d(1, ch // 4  , 4, stride=4, bias=True, padding=0),
#             nn.GELU(),
#             nn.Unflatten(2, (grid_dim, grid_dim)),
#             nn.Conv2d(ch // 4, ch, 3, 1, 1)
#         )
#         layer_adapter.to(device)
#         self.input_adapters.append(layer_adapter)
        
#         ch = channels[1]
#         grid_dim = grid_dims[1]
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, ch // 4 , 17, stride=1, bias=True, padding=8),
#             nn.GELU(),
#             nn.Unflatten(2, (grid_dim, grid_dim)),
#             nn.Conv2d(ch // 4, ch, 3, 1, 1)
#         )
#         layer_adapter.to(device)
#         self.input_adapters.append(layer_adapter)
        
#         ch = channels[2]
#         grid_dim = grid_dims[2]
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, ch // 4 , 17, stride=4, bias=True, padding=8),
#             nn.GELU(),
#             nn.Unflatten(2, (grid_dim, grid_dim)),
#             nn.Conv2d(ch // 4, ch, 3, 1, 1)
#         )
#         layer_adapter.to(device)
#         self.input_adapters.append(layer_adapter)
    
#         ch = channels[3]
#         grid_dim = grid_dims[3]
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, ch // 4, 17, stride=16, bias=True, padding=2),
#             nn.GELU(),
#             nn.Unflatten(2, (grid_dim, grid_dim)),
#             nn.Conv2d(ch // 4, ch, 3, 1, 1)
#         )
#         layer_adapter.to(device)
#         self.input_adapters.append(layer_adapter)
        
#         ############### MIDDLE BLOCKS ##############################

#         ch = channels[3]
#         grid_dim = grid_dims[3]
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, ch // 4, 17, stride=16, bias=True, padding=2),
#             nn.GELU(),
#             nn.Unflatten(2, (grid_dim, grid_dim)),
#             nn.Conv2d(ch // 4, ch, 3, 1, 1)
#         )
#         layer_adapter.to(device)
#         self.middle_adapters.append(layer_adapter)
        
#         ############### OUTPUT BLOCKS ##############################
        
#         ch = channels[3]
#         grid_dim = grid_dims[3]
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, ch // 4, 17, stride=16, bias=True, padding=2),
#             nn.GELU(),
#             nn.Unflatten(2, (grid_dim, grid_dim)),        x_init = self.input_adapters[0](x)   
#         for module in self.input_adapters[1:]:
#             adapter_out = module(x_init)
#             input_features.append(adapter_out)    
            
#         x_init = self.middle_adapters[0](x)   
#         for module in self.middle_adapters[1:]:
#             adapter_out = module(x_init)
#             middle_features.append(adapter_out)
#             nn.Conv2d(ch // 4, ch, 3, 1, 1)
#         )
#         layer_adapter.to(device)
#         self.output_adapters.append(layer_adapter)
        
        
#         ch = channels[2]
#         grid_dim = grid_dims[2]
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, ch // 4 , 17, stride=4, bias=True, padding=8),
#             nn.GELU(),
#             nn.Unflatten(2, (grid_dim, grid_dim)),
#             nn.Conv2d(ch // 4, ch, 3, 1, 1)
#         )
#         layer_adapter.to(device)
#         self.output_adapters.append(layer_adapter)

#         ch = channels[1]
#         grid_dim = grid_dims[1]
#         layer_adapter = nn.Sequential(
#             nn.Conv1d(1, ch // 4 , 17, stride=1, bias=True, padding=8),
#             nn.GELU(),
#             nn.Unflatten(2, (grid_dim, grid_dim)),
#             nn.Conv2d(ch // 4, ch, 3, 1, 1)
#         )
#         layer_adapter.to(device)
#         self.output_adapters.append(layer_adapter)
        
#         ch = channels[0]
#         grid_dim = grid_dims[0]
#         layer_adapter = nn.Sequential(
#             nn.ConvTranspose1d(1, ch // 4  , 4, stride=4, bias=True, padding=0),
#             nn.GELU(),
#             nn.Unflatten(2, (grid_dim, grid_dim)),
#             nn.Conv2d(ch // 4, ch, 3, 1, 1)
#         )
#         layer_adapter.to(device)
#         self.output_adapters.append(layer_adapter)
        
>>>>>>> 31e0cee6b294499a442e056c919926632ffd3099
        ################ previous adapter implementation with mlp layers 
        
#         # for input blocks
#         for ind, (ch, grid_dim) in enumerate(zip(channels, grid_dims)):
#             print(f" ch {ch} and {grid_dim}")
#             layer_adapter = nn.Sequential(
#                 nn.Linear(adapter_input_dim, ch, bias=False),
#                 nn.GELU(),
#                 nn.Linear(ch, ch, bias=False))
#             layer_adapter.to(device)
#             self.input_adapters.append(layer_adapter)
        
#         # for middle block 
#         layer_adapter = nn.Sequential(
#             nn.Linear(adapter_input_dim, channels[-1], bias=False),
#             nn.GELU(),
#             nn.Linear(channels[-1], channels[-1], bias=False))
#         layer_adapter.to(device)
#         self.middle_adapters.append(layer_adapter)
        
#         # for output blocks
#         for ind, (ch, grid_dim)  in enumerate(zip(reversed(channels),reversed(grid_dims))):
#             print(f" ch {ch} and {grid_dim}")

#             layer_adapter = nn.Sequential(
#                 nn.Linear(adapter_input_dim, ch, bias=False),
#                 nn.GELU(),
#                 nn.Linear(ch, ch, bias=False))
#             layer_adapter.to(device)
#             self.output_adapters.append(layer_adapter)
        
        self.model = nn.ModuleList(self.input_adapters + self.middle_adapters + self.output_adapters)

    def forward(self, x):
        input_features = []
        middle_features = []
        output_features = []
        
#         for i in range(len(self.channels)):
#             adapter_out = self.input_adapters[i](x)
#             input_features.append(adapter_out)

#         x_init = self.input_adapters[0](x)   
#         for module in self.input_adapters[1:]:
#             adapter_out = module(x_init)
#             input_features.append(adapter_out)    
            
#         x_init = self.middle_adapters[0](x)   
#         for module in self.middle_adapters[1:]:
#             adapter_out = module(x_init)
#             middle_features.append(adapter_out)

        x_init = self.output_adapters[0](x)   
        for module in self.output_adapters[1:]:
            adapter_out = module(x_init)
            output_features.append(adapter_out)
            
#         for module in self.input_adapters:
#             adapter_out = module(x)
#             input_features.append(adapter_out)  
    
#         for module in self.middle_adapters:
#             adapter_out = module(x)
#             middle_features.append(adapter_out)
            
<<<<<<< HEAD
        for module in self.output_adapters:
            adapter_out = module(x)
<<<<<<< HEAD
            output_features.append(adapter_out)
            
        return input_features , middle_features, output_features
=======
            output_features.append(adapter_out) 
=======
#         for module in self.output_adapters:
#             adapter_out = module(x)
#             output_features.append(adapter_out) 
>>>>>>> 228b50c1afa4e32e77d268d2b93b550456490c33


   
<<<<<<< HEAD
        return input_features, middle_features, output_features
>>>>>>> 31e0cee6b294499a442e056c919926632ffd3099
=======
        return None, None, output_features
>>>>>>> 662e7fce64ab7f82788b4e45601f173b2aac73e1
    
            