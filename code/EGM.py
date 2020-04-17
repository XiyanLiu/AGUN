import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv_Block, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.conv_2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.conv_2(x)
        return x

class EGM(nn.Module):

    def __init__(self, in_dim):
        super(EGM, self).__init__()
        self.conv_block_1 = Conv_Block(in_dim, in_dim // 8)
        self.conv_block_2 = Conv_Block(in_dim, in_dim // 8)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, F_input):
        '''
        :param F_input: tensor of size batchsize * C * H * W
        :return: tensor of size batchsize * C * H * W
        '''
        bs, C, H, W = F_input.size()
        conv_out_1 = self.conv_block_1(F_input).view(bs, -1, H * W)    # batchsize * (C/8) * (H*W)
        conv_out_1 = conv_out_1.permute(0, 2, 1)                       # transpose: batchsize * (H*W) * (C/8)
        conv_out_2 = self.conv_block_2(F_input).view(bs, -1, H * W)    # batchsize * (C/8) * (H*W)
        mm_1 = torch.bmm(conv_out_1, conv_out_2)                       # Result of matrix multiplication: batchsize * (H*W) * (H*W)
        S = self.softmax(mm_1)                                         # batchsize * (H*W) * (H*W)
        F_reshape = F_input.view(bs, -1, H * W)                        # Reshape: batchsize * C * H * W -> batchsize * C * (H*W)
        mm_2 = torch.bmm(F_reshape, S.permute(0, 2, 1))                # Result of matrix multiplication between F and S: batchsize * C * (H*W)
        mm_2 = mm_2.view(bs, C, H, W)                                  # Reshape: batchsize * C * (H*W) -> batchsize * C * H * W
        F_output = F_input + mm_2                                      # Output: batchsize * C * H * W

        return F_output

if __name__ == '__main__':
    ### only for test ###
    BS = 2
    C = 32
    H = 64
    W = 64
    data_input = torch.rand(BS, C, H, W)
    print('data_input:', data_input, data_input.size())
    EGM_func = EGM(C)
    data_output = EGM_func(data_input)
    print('data_output:', data_output, data_output.size())
