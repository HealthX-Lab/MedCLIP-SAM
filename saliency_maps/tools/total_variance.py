import torch

class TotalVariance(torch.nn.Module):
    def __init__(self, p: int = 1, q: int = 1):
        super(TotalVariance, self).__init__()
        self.p = p
        self.q = q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, num_channels, height, width)
        :return: (batch_size, 1)
        """
        if len(x.size()) != 4:
            if len(x.size()) == 3:
                x = x.unsqueeze(0) # (1, num_channels, height, width)
            elif len(x.size()) == 2:
                x = x.unsqueeze(0).unsqueeze(0) # (1, 1, height, width)
            else:
                raise ValueError("Invalid number of dimensions.")
        pixel_diff_hirozontal = x[:, :, :, 1:] - x[:, :, :, :-1]
        pixel_diff_vertical = x[:, :, 1:, :] - x[:, :, :-1, :]
        
        tv_h = torch.sum(torch.pow(pixel_diff_hirozontal, self.p).abs(), dim=(2, 3)).pow(self.q/self.p)
        tv_v = torch.sum(torch.pow(pixel_diff_vertical, self.p).abs(), dim=(2, 3)).pow(self.q/self.p)
        return tv_h + tv_v
    
    