import torch
from torch import nn
from torch.nn import ModuleList


class net(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer[0]
        self.res_seq = layer[1]

        self.MSE = nn.MSELoss()
        self.CEB = nn.CrossEntropyLoss()
        self.BCE = nn.BCELoss()

    def forward(self, x):
        outputs = [x]
        for i, layer in enumerate(self.layer):
            out = layer(outputs[-1])

            for res in self.res_seq:
                if res[1] == i:
                    # out += outputs[res[0]]
                    out = torch.cat((out, outputs[res[0]]), dim=-1)
            outputs.append(out)
        return outputs[-1]


if __name__ == '__main__':
    # layer = ModuleList()
    # layer.append(nn.Linear(10, 10))
    # layer.append(nn.Linear(10, 10))
    # layer.append(nn.Linear(10, 10))
    # layer.append(nn.Linear(10, 10))
    # layer.append(nn.Linear(20, 10))
    # layer.append(nn.Linear(10, 10))
    # layer = (layer, [(0, 3)])
    # net = net(layer)
    # x = torch.rand(([1, 10]))
    # print(net(x).shape)
    out_list = [(2),[(torch.rand([1,2,3]),torch.rand([1,2,3])),((torch.rand([1,2,3]),torch.rand([1,3])))]]
    print( [y.shape for x in out_list[1] for y in x])