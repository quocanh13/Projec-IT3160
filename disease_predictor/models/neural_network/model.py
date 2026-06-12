import torch as tc
import torch.nn as nn

class DiseasePredictor(nn.Module):
    def __init__(
        self, 
        input_size = 377, 
        output_size = 773
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024, True), nn.ReLU(inplace=True),
            nn.Linear(1024, 512, True), nn.ReLU(inplace=True),
            nn.Linear(512, 256, True), nn.ReLU(inplace=True),
            nn.Linear(256, output_size, True)
        )
        
    def forward(
        self, 
        X: tc.Tensor
    ) -> tc.Tensor:
        return self.model(X)
    
    def predict_proba(
        self, 
        X: tc.Tensor
    ) -> tc.Tensor:
        logits = self.model(X)
        softmax = nn.Softmax(dim=1)
        return softmax(logits)
    
    def save_state(self, path: str = "./disease_predictor/models/neural_network/state.pth"):
        tc.save(self.state_dict(), path)
    
    def load_state(self, path: str = "./disease_predictor/models/neural_network/state.pth"):
        device = "cuda" if tc.cuda.is_available() else "cpu"
        self.load_state_dict(tc.load(path, map_location=device))
    
    