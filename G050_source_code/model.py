from torchvision import models
from torchvision.models import vit_h_14,ViT_H_14_Weights
from torchvision.models.video import swin3d_t,Swin3D_T_Weights
import torch
import torch.nn as nn
from custom_model_LSTM import CNNLSTM

def build_model(fine_tune=True, num_classes=2):
    # model = models.video.mc3_18(weights='DEFAULT')
    model=swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
    # model=CNNLSTM(num_classes=2)
    # model=CNNLSTM(num_classes=2)
    # model.resnet.requires_grad=False
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    if not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # model.fc = nn.Linear(in_features=512, out_features=num_classes)
    model.head = torch.nn.Linear(model.head.in_features, num_classes)
    return model

if __name__ == '__main__':
    model = build_model()
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
