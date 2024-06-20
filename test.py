from torch import nn

if __name__ == '__main__':
    a = "[16,32,32]"
    print(eval(a), type(eval(a)))
    a = nn.BatchNorm2d(eval(a))
    print(a)
