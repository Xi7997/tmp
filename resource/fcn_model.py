import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models



class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()

        # Load the pretrained VGG-16 and use its features
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())

        # Encoder
        self.features_block1 = nn.Sequential(*features[:5])  # First pooling
        self.features_block2 = nn.Sequential(*features[5:10])  # Second pooling
        self.features_block3 = nn.Sequential(*features[10:17])  # Third pooling
        self.features_block4 = nn.Sequential(*features[17:24])  # Fourth pooling
        self.features_block5 = nn.Sequential(*features[24:])  # Fifth pooling

        # Modify the classifier part of VGG-16
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        # Decoder
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_final = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

        # Skip connections
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)





    def forward(self, x):
        # Encoder
        block1 = self.features_block1(x)
        block2 = self.features_block2(block1)
        block3 = self.features_block3(block2)
        block4 = self.features_block4(block3)
        block5 = self.features_block5(block4)

        # Classifier
        score = self.classifier(block5)

        # Decoder
        upscore2 = self.upscore2(score)
        score_pool4 = self.score_pool4(block4)
        if upscore2.size()[2:] != score_pool4.size()[2:]:
            upscore2 = F.interpolate(upscore2, score_pool4.size()[2:], mode='bilinear', align_corners=True)
        upscore_pool4 = self.upscore_pool4(upscore2 + score_pool4)
        score_pool3 = self.score_pool3(block3)
        if upscore_pool4.size()[2:] != score_pool3.size()[2:]:
            upscore_pool4 = F.interpolate(upscore_pool4, score_pool3.size()[2:], mode='bilinear', align_corners=True)

        upscore_final = self.upscore_final(upscore_pool4 + score_pool3)

        return upscore_final

