import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels,
                              kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C, H, W)
        Returns
        -------
        torch.Tensor
            of dimensions (B, C, H, W)
        """
        c = self.conv(x)
        b = self.bn(c)
        r = self.relu(b)
        return r



class CaptchaModel(nn.Module):
   def __init__(self, num_chars):
      super(CaptchaModel, self).__init__()
      self.conv_1 = ConvBlock(3, 128)
      self.conv_2 = ConvBlock(128, 64)
      self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
      
      self.linear_1 = nn.Linear(1152, 64)
      self.drop_1 = nn.Dropout(0.2)
      
      self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25)
      self.output = nn.Linear(64, num_chars +1)
   
   def forward(self, images, targets=None):
      bs, c, h, w = images.size()
      
      x = self.conv_1(images)
      x = self.max_pool(x)
      x = self.conv_2(x)
      x = self.max_pool(x) # -> 1, 64, 18, 75
      
      x = x.permute(0, 3, 1, 2) # -> 1, 75, 64, 18
      x = x.view(bs, x.size(1), -1) # -> 1, 75, 64*18
      x = self.linear_1(x) # -> 1, 75, 64
      x = self.drop_1(x)
      
      x, _ = self.gru(x) # -> 1, 75, 64
      x = self.output(x) # -> 1, 75, 64
      x = x.permute(1, 0, 2) # -> 75, 1, 64
      if targets is not None:
         log_softmax_val = F.log_softmax(x, 2)
         input_lengths = torch.full(size=(bs, ), 
                                    fill_value=log_softmax_val.size(0),
                                    dtype=torch.int32
                                    )
         target_lengths = torch.full(size=(bs, ),
                                    fill_value=targets.size(1),
                                    dtype=torch.int32
                                    )
         loss = nn.CTCLoss(blank=0)(
            log_softmax_val, targets, input_lengths, target_lengths
            )
         return x, loss  
      return x, None 

if __name__ == '__main__':
   cm = CaptchaModel(19)
   img = torch.rand(1, 3, 75, 300)
   target = torch.randint(1, 20, (1, 5))
   x, loss = cm(img, target)
      
      
      
