import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
import torch

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
       
    def forward(self, x_3d):
        hidden = None

        # Iterate over each frame of a video in a video of batch * frames * channels * height * width
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t])  
            # Pass latent representation of frame through lstm and update hidden state
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        # Get the last hidden state (hidden is a tuple with both hidden and cell state in it)
        x = self.fc1(hidden[0][-1])
        x = F.relu(x)
        x = self.fc2(x)

        return x

# import torch.nn as nn
# import torch
# import torch.nn.functional as F
# from torchvision.models.video import r3d_18

# class Conv3DLSTM(nn.Module):
#     def __init__(self, num_classes=2, hidden_size=256, num_layers=2):
#         super(Conv3DLSTM, self).__init__()
#         self.r3d = r3d_18(pretrained=True)
#         self.r3d.fc = nn.Identity()

#         # for param in self.r3d.parameters():
#         #     param.requires_grad = False
        
#         self.lstm = nn.LSTM(
#             input_size=512,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True
#         )
        
#         self.fc1 = nn.Linear(hidden_size, 128)
#         self.fc2 = nn.Linear(128, num_classes)
        
#     def forward(self, x):
#         # Pass the input through the 3D CNN
#         x = self.r3d(x)
        
#         # Reshape the output to (batch_size, sequence_length, feature_size)
#         x = x.view(x.size(0), -1, x.size(1))
        
#         # Pass the sequence through the LSTM
#         x, _ = self.lstm(x)
        
#         # Take the last hidden state of the LSTM
#         x = x[:, -1, :]
        
#         # Pass through the fully connected layers
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
        
#         return x