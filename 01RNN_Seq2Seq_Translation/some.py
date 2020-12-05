import torch

# a = torch.zeros((3, 5))
# print(a)
b = torch.randn((1, 5))
x, y = b.data.topk(1)
print(y.squeeze())
# a.index_copy_(dim=0, index=torch.LongTensor([0, 1]), source=b)
# print(a)
# print(torch.arange(0, 3, dtype=torch.long,de))
