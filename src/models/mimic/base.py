import torch
from torch import nn

'''
class SeqWithAux(nn.Module):
    def __init__(self, modules, aux_idx, aux_input_channel, aux_output_size):
        super().__init__()
        self.head_modules = nn.Sequential(*modules[:aux_idx + 1])
        self.aux_seq = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(aux_input_channel, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.Conv2d(128, 768, kernel_size=5, bias=False),
            nn.BatchNorm2d(768, eps=0.001),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.linear = nn.Linear(768, aux_output_size)
        self.tail_modules = nn.Sequential(*modules[aux_idx + 1:])

    def forward(self, sample_batch):
        zs = self.head_modules(sample_batch)
        if self.training:
            zs_aux = self.aux_seq(zs)
            return self.tail_modules(zs), self.linear(zs_aux.view(zs_aux.size(0), -1))
        return self.tail_modules(zs)
'''

class BaseHeadMimic(nn.Module):
    def __init__(self, input_size=224):
        super().__init__()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, sample_batch):
        raise NotImplementedError('forward function must be implemented')

    def forward_to_bn(self, sample_batch):
        raise NotImplementedError('forward function must be implemented')

    def forward_from_bn(self, sample_batch):
        raise NotImplementedError('forward function must be implemented')

    def bn_shape(self, input_shape, device):
        self.eval()
        with torch.no_grad():
            input_ = torch.empty(1, input_shape[0], input_shape[1], input_shape[2], device=device)
            bn_output = self.forward_to_bn(input_)
            return bn_output.shape[1:]


class BaseMimic(nn.Module):
    def __init__(self, head, tail):
        """

        Args:
            head (BaseHeadMimic):
            tail (list of nn.Module):
        """
        super().__init__()
        self.head = head
        self.tail = nn.Sequential(*tail[:-1])
        self.classifier = tail[-1]
        self.device = None

    def forward(self, sample_batch):
        zs = sample_batch
        if self.head is not None:
            zs = self.head(zs)
        zs = self.tail(zs)
        return self.classifier(zs.view(zs.size(0), 0 if zs.size(0) == 0 else -1))

    def forward_to_bn(self, sample_batch):
        return self.head.forward_to_bn(sample_batch)

    def forward_from_bn(self, sample_batch):
        zs = self.head.forward_from_bn(sample_batch)
        zs = self.tail(zs)
        return self.classifier(zs.view(zs.size(0), 0 if zs.size(0) == 0 else -1))

    @property
    def out_features(self):
        return self.classifier.out_features

