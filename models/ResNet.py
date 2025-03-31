import torch.nn as nn
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self,args):
        super(ResNet18, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.fcrelu = nn.ReLU()
        self.fc1 = nn.Linear(1000, args.num_classes)

    def forward(self, x):
        y = self.backbone(x)
        y = self.fcrelu(y)
        y = self.fc1(y)
        return y

if __name__=="__main__":
    import sys
    sys.path.append('/home/aaa/yangjie/Fed_win/')
    from utils.options import args_parser
    args = args_parser()
    ourModel = ResNet18(args)
    print(ourModel)
    
    