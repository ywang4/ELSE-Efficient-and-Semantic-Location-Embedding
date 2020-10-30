import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiLabelClassifer(nn.Module):
    def __init__(self, embedding_net, num_class, embedding_size):
        super(MultiLabelClassifer, self).__init__()
        self.num_class = num_class
        self.embedding_size = embedding_size
        self.embedding_net = embedding_net
        self.fc = nn.Linear(self.embedding_size, self.num_class)

    def forward(self, x):
        x = self.embedding_net(x)
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.embedding_net(x)


class MultilabelEmbeddingNet(nn.Module):
    def __init__(self, embedding_size=8):
        super(MultilabelEmbeddingNet, self).__init__()

        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 9, kernel_size=5)

        self.conv1_bn = nn.BatchNorm2d(9)
        self.conv2 = nn.Conv2d(9, 18, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(18)
        self.conv3 = nn.Conv2d(18, 36, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(36)
        self.fc1 = nn.Linear(22500, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, self.embedding_size)
        self.fc3_bn = nn.BatchNorm1d(self.embedding_size)

        # self._initialize()

    # def _initialize(self):
    #    if hasattr(self, 'l_y'):
    #        init.xavier_uniform_(self.l_y.weight.data)

    def forward(self, x):  # input bx24x332  bx3x224x224
        # x = x.unsqueeze(1)  # input bx1x24x332
        x = x.float()
        x = F.avg_pool2d(F.selu(self.conv1_bn(self.conv1(x))), 2)  # input bx4x10x164  bx12x110x110
        x = F.avg_pool2d(F.selu(self.conv2_bn(self.conv2(x))), 2)  # input bx16x3x80  bx48x53x53
        # x = F.dropout(x, p=0.1, training=self.training)
        x = F.avg_pool2d(F.selu(self.conv3_bn(self.conv3(x))), 2)  # bx96x26x26
        # x = F.selu(self.conv2(x)).squeeze(2)  # input bx32x78  bx96x676
        x = x.view(-1, self.num_flat_features(x))  # input bx2496  bx64896
        x = F.selu(self.fc1_bn(self.fc1(x)))  # bx3
        x = F.selu(self.fc2_bn(self.fc2(x)))  # bx3
        x = F.selu(self.fc3_bn(self.fc3(x)))  # bx3
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Resent18EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=512, pretrained=True):
        super(Resent18EmbeddingNet, self).__init__()
        self.pretrained = pretrained
        self.embedding_size = embedding_size

        self.model = models.resnet18(pretrained=self.pretrained)
        self.resnet = nn.Sequential(*list(self.model.children())[:-1])

        self.embedding_layer = nn.Linear(self.model.fc.in_features, self.embedding_size)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.embedding_layer(x)
        x = F.selu(x)
        return x


class Resent50EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=2048, pretrained=True):
        super(Resent50EmbeddingNet, self).__init__()
        self.pretrained = pretrained
        self.embedding_size = embedding_size

        self.model = models.resnet50(pretrained=self.pretrained)
        self.resnet = nn.Sequential(*list(self.model.children())[:-1])

        self.embedding_layer = nn.Linear(self.model.fc.in_features, self.embedding_size)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = F.selu(self.embedding_layer(x))
        return x


class Densenet121EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=512, pretrained=True):
        super(Densenet121EmbeddingNet, self).__init__()
        self.pretrained = pretrained
        self.embedding_size = embedding_size

        self.densenet = models.densenet121(pretrained=self.pretrained)
        self.densenet.classifier = nn.Linear(1024, embedding_size)

    def forward(self, x):
        x = self.densenet(x)
        x = F.selu(x)
        return x
