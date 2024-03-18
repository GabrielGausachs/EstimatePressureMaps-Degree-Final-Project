import torch
class pwrsWtL(torch.nn.Module):
    def __init__(self, type_L='L2'):
        super(pwrsWtL, self).__init__()
        # no clip, no base wht, wht outside
        self.type_L = type_L

    def forward(self, src, tar, wht=1):     # only work on 0,1
        # if self.clipMod == 'clip11':
        #     wht = (tar + 1) / 2 * self.whtScal + self.baseWht
        # elif self.clipMod == 'clip01':
        #     wht = tar * self.whtScal + self.baseWht
        # else:  # no processing only adds wht in
        #     wht = tar + self.baseWht
        if 'L1' == self.type_L:
            loss = (torch.abs(src - tar) * wht).mean()  # L1 loss
        elif 'L2' == self.type_L:
            loss = (wht * (src - tar) ** 2).mean()  # MSE loss, mean value, so only scale down. expect device cpu?
        else:
            print("no such loss implementation", self.type_L)
        return loss