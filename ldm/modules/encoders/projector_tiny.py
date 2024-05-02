import torch

class AudioMLP (torch.nn.Module):
    def __init__(self, input_dim = 1024, output_dim = 768):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.projector = torch.nn.Sequential(torch.nn.Linear(self.input_dim, self.output_dim), 
        torch.nn.GELU(),
        torch.nn.Linear(self.output_dim, self.output_dim), 
        ).to("cuda")

    def forward(self, x):
        return self.projector(x)