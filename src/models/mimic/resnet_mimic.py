from torch import nn
from compressai.layers import GDN1

from models.mimic.base import BaseHeadMimic, BaseMimic


class BottleneckIdx:
    V1 = 4
    V2 = 4
    V3 = 4
    V4 = 7
    V5 = 10


def mimic_version1(make_bottleneck, bottleneck_channel):
    if make_bottleneck:
        return nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channel, 256, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
    return nn.Sequential(
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=1)
    )


def mimic_version2b_with_aux(modules, aux_idx, bottleneck_channel, aux_output_size=1000):
    # return SeqWithAux(modules, aux_idx=aux_idx, aux_input_channel=bottleneck_channel, aux_output_size=aux_output_size)
    return nn.Sequential(*modules)

def mimic_version2(make_bottleneck, dataset_name, bottleneck_channel, use_aux):
    if make_bottleneck:
        modules = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.Conv2d(bottleneck_channel, 64, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
        aux_idx = 2
        aux_output_size = 101
        if dataset_name in ['imagenet', 'cifar100']:
            modules = nn.Sequential(
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(bottleneck_channel),
                nn.ReLU(inplace=True)
            ), nn.Sequential(
                nn.Conv2d(bottleneck_channel, 512, kernel_size=2, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=1)
            )
            aux_output_size = 1000
        # return mimic_version2b_with_aux(modules, aux_idx, bottleneck_channel, aux_output_size) if use_aux \
        #   else nn.Sequential(*modules)
        return modules
    return nn.Sequential(
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


def mimic_version3(make_bottleneck, bottleneck_channel):
    if make_bottleneck:
        return nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.Conv2d(bottleneck_channel, 64, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 2048, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=7, stride=2)
        )
    return nn.Sequential(
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=2, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=2, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=2, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 1024, kernel_size=2, stride=1, bias=False),
        nn.BatchNorm2d(1024),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 2048, kernel_size=2, stride=1, bias=False),
        nn.BatchNorm2d(2048),
        nn.ReLU(inplace=True),
        nn.Conv2d(2048, 2048, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=7, stride=2)
    )


def mimic_version4(make_bottleneck, bottleneck_channel):
    if make_bottleneck:
        return nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, bottleneck_channel, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(30, mode='bilinear'),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
    return nn.Sequential(
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 128, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=1)
    )


def mimic_version5(make_bottleneck, bottleneck_channels):
    if make_bottleneck:
        return nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, bottleneck_channels, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        ), nn.Sequential(
            nn.Conv2d(bottleneck_channels, 512, kernel_size=2, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
    return nn.Sequential(
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


def mimic_version6(make_bottleneck, bottleneck_channels):
    if make_bottleneck:
        return nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, bottleneck_channels, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        ), nn.Sequential(
            nn.Conv2d(bottleneck_channels, 32, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
    return nn.Sequential(
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

def mimic_version7(bottleneck_channels, target_channels=512):
    return nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1, bias=False),
        GDN1(64),
        nn.Conv2d(64, 32, kernel_size=2, stride=1, padding=0, bias=False),
        GDN1(32),
        nn.Conv2d(32, bottleneck_channels, kernel_size=2, stride=2, padding=1, bias=False),
        GDN1(bottleneck_channels),
        #nn.BatchNorm2d(bottleneck_channels),
        #nn.ReLU(inplace=True),
    ), nn.Sequential(
        nn.Conv2d(bottleneck_channels, 32, kernel_size=2, stride=1, padding=1, bias=False),
        GDN1(32),
        #nn.Conv2d(32, round(target_channels/8), kernel_size=3, stride=1, padding=1, bias=False),
        #GDN1(64),
        nn.Conv2d(round(target_channels/16), round(target_channels/4), kernel_size=2, stride=1, padding=1, bias=False),
        GDN1(128),
        nn.Conv2d(round(target_channels/4), round(target_channels/2), kernel_size=2, stride=1, bias=False),
        GDN1(256),
        nn.Conv2d(round(target_channels/2), target_channels, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=1)
    )

# new Matsubara paper
def mimic_version8(bottleneck_channels, target_channels=512):
    return nn.Sequential(
        #nn.Conv2d(3, bottleneck_channels * 4, kernel_size=5, stride=2, padding=2, bias=False),
        nn.Conv2d(3, bottleneck_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
        GDN1(bottleneck_channels * 4),
        nn.Conv2d(bottleneck_channels * 4, bottleneck_channels * 3, kernel_size=4, stride=2, padding=1, bias=False),
        GDN1(bottleneck_channels * 3),
        nn.Conv2d(bottleneck_channels * 3, bottleneck_channels * 2, kernel_size=2, stride=1, padding=0, bias=False),
        GDN1(bottleneck_channels * 2),
        nn.Conv2d(bottleneck_channels * 2, bottleneck_channels, kernel_size=2, stride=1, padding=0, bias=False)
    ), nn.Sequential(
        nn.Conv2d(bottleneck_channels, target_channels * 2, kernel_size=2, stride=1, padding=1, bias=False),
        GDN1(target_channels * 2, inverse=True),
        nn.Conv2d(target_channels * 2, target_channels, kernel_size=2, stride=1, padding=0, bias=False),
        GDN1(target_channels, inverse=True),
        nn.Conv2d(target_channels, target_channels, kernel_size=2, stride=1, padding=1, bias=False),
        GDN1(target_channels, inverse=True),
        nn.Conv2d(target_channels, target_channels, kernel_size=2, stride=1, padding=1, bias=False)
    )


class ResNetHeadMimic(BaseHeadMimic):
    # designed for input image size [3, 224, 224]
    def __init__(self, version, dataset_name, bottleneck_channels=3, use_aux=False, input_size=224, adapt_size=True):
        super().__init__()
        xtr_k, xtr_s, xtr_p = 7, 2, 3
        if adapt_size and input_size == 32:
            xtr_k, xtr_s, xtr_p = 3, 1, 1
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=xtr_k, stride=xtr_s, padding=xtr_p, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if input_size != 32 else nn.Identity()
        )
        if version in ['1', '1b']:
            self.module_seq1, self.module_seq2 = mimic_version1(version == '1b', bottleneck_channels)
        elif version in ['2', '2b']:
            self.module_seq1, self.module_seq2 = mimic_version2(version == '2b', dataset_name, bottleneck_channels, use_aux)
        elif version in ['3', '3b']:
            self.module_seq1, self.module_seq2 = mimic_version3(version == '3b', bottleneck_channels)
        elif version in ['5', '5b']:
            self.module_seq1, self.module_seq2 = mimic_version5(version == '5b', bottleneck_channels)
        elif version in ['6', '6b']:
            self.module_seq1, self.module_seq2 = mimic_version6(version == '6b', bottleneck_channels)
        elif version in ['7', '7b']:
            self.module_seq1, self.module_seq2 = mimic_version7(bottleneck_channels)
        elif version in ['8', '8b']:
            self.extractor = nn.Identity()
            self.module_seq1, self.module_seq2 = mimic_version8(bottleneck_channels)
        else:
            raise ValueError('version `{}` is not expected'.format(version))
        self.initialize_weights()

    def forward(self, sample_batch):
        zs = self.extractor(sample_batch)
        zs = self.module_seq1(zs)
        return self.module_seq2(zs)

    def forward_to_bn(self, sample_batch):
        zs = self.extractor(sample_batch)
        return self.module_seq1(zs)

    def forward_from_bn(self, sample_batch):
        return self.module_seq2(sample_batch)

    def freeze_encoder(self):
        for p in self.extractor.parameters():
            p.requires_grad = False
        for p in self.module_seq1.parameters():
            p.requires_grad = False


class ResNetHeadMimic_32(BaseHeadMimic):
    # designed for input image size [3, 32, 32]
    def __init__(self, version, dataset_name, bottleneck_channels=3, use_aux=False):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        if version in ['1', '1b']:
            self.module_seq1, self.module_seq2 = mimic_version1(version == '1b', bottleneck_channels)
        elif version in ['2', '2b']:
            self.module_seq1, self.module_seq2 = mimic_version2(version == '2b', dataset_name, bottleneck_channels, use_aux)
        elif version in ['3', '3b']:
            self.module_seq1, self.module_seq2 = mimic_version3(version == '3b', bottleneck_channels)
        elif version in ['4', '4b']:
            self.module_seq1, self.module_seq2 = mimic_version4(version == '4b', bottleneck_channels)
        elif version in ['5', '5b']:
            self.module_seq1, self.module_seq2 = mimic_version5(version == '5b', bottleneck_channels)
        elif version in ['6', '6b']:
            self.module_seq1, self.module_seq2 = mimic_version6(version == '6b', bottleneck_channels)
        else:
            raise ValueError('version `{}` is not expected'.format(version))
        self.initialize_weights()

    def forward(self, sample_batch):
        zs = self.extractor(sample_batch)
        zs = self.module_seq1(zs)
        return self.module_seq2(zs)

    def forward_to_bn(self, sample_batch):
        zs = self.extractor(sample_batch)
        return self.module_seq1(zs)

    def forward_from_bn(self, sample_batch):
        return self.module_seq2(sample_batch)


class ResNetMimic(BaseMimic):
    def __init__(self, head, tail):
        super().__init__(head, tail)

    def forward(self, sample_batch):
        return super().forward(sample_batch)
