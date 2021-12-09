import torch
from torch import nn

from models.mimic.base import BaseHeadMimic, BaseMimic


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input_: torch.Tensor):
        """
        Reshapes the input according to the shape saved in the view data structure.
        """
        out = input_.view(self.shape)
        return out


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'Flatten()'

    def forward(self, input_: torch.Tensor):
        """
        Reshapes the input flattening it (preserving batch dimension)
        """
        out = input_.flatten(1)
        return out


def get_encoder_decoder(units=1744, n_z=1024, in_size=32, out_size=16, c_out=512):

    dim_h = int(units/2**3)
    dim_g = int(c_out/2**3)
    z_size = out_size / 2**3    # could this be used to determine kernel sizes for the decoder?

    return nn.Sequential(
        nn.Conv2d(3, dim_h, kernel_size=4, stride=2, padding=1, bias=False),
        nn.ReLU(True),
        nn.Conv2d(dim_h, dim_h * 2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(dim_h * 2),
        nn.ReLU(True),
        nn.Conv2d(dim_h * 2, dim_h * 4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(dim_h * 4),
        nn.ReLU(True),
        nn.Conv2d(dim_h * 4, dim_h * 8, kernel_size=4, bias=False),
        nn.BatchNorm2d(dim_h * 8),
        nn.ReLU(True),
        Flatten(),
        #nn.Linear(dim_h * (2 ** 3), n_z)
    ), nn.Sequential(
        #nn.Linear(n_z, dim_h * (2 ** 3) * 4 * 4),
        #nn.ReLU(),
        #View([-1, dim_h * (2 ** 3), 4, 4]),
        View([-1, int(dim_h/2), 4, 4]),
        #nn.ConvTranspose2d(dim_h * (2 ** 3), dim_g * (2 ** 1), kernel_size=2),
        nn.ConvTranspose2d(int(dim_h/2), dim_g * (2 ** 1), kernel_size=2),
        nn.BatchNorm2d(dim_g * (2 ** 1)),
        nn.ReLU(True),
        nn.ConvTranspose2d(dim_g * (2 ** 1), dim_g * (2 ** 2), kernel_size=3),
        nn.BatchNorm2d(dim_g * (2 ** 2)),
        nn.ReLU(True),
        nn.ConvTranspose2d(dim_g * (2 ** 2), dim_g * (2 ** 3), kernel_size=4, stride=2),
        nn.Sigmoid()
    )


class DCGANHeadMimic(BaseHeadMimic):
    # designed for input image size [3, 224, 224]
    def __init__(self, version, dataset_name, bottleneck_channels=3, use_aux=False, input_size=224):
        super().__init__()
        self.extractor = nn.Identity()
        self.module_seq1, self.module_seq2 = get_encoder_decoder()
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


class DCGANMimic(BaseMimic):
    def __init__(self, head, tail):
        super().__init__(head, tail)

    def forward(self, sample_batch):
        return super().forward(sample_batch)
