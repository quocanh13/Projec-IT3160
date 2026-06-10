import torch as tc
import torch.nn as nn

class DiseasePredictor(nn.Module):
    def __int__(
        self, 
        input_size = 377, 
        output_size = 773
    ):
        self.model = nn.Sequential(
            nn.Linear(input_size, 512, True),
            nn.Linear(256, 128, True),
            nn.Linear(128, 10, True)
        )
        
    def forward(
        self, 
        X: tc.Tensor
    ) -> tc.Tensor:
        