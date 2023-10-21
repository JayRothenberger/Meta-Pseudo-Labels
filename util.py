import torch


def accuracy(output, target, topk=(1,)):
    output = output.to(torch.device('cpu'))
    target = target.to(torch.device('cpu'))
    maxk = max(topk)
    batch_size = target.shape[0]

    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, maxk).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class RepeatLoader:
  def __init__(self, loader):
      self.loader = loader
      self.it = iter(loader)
      
  def __iter__(self):
    return self

  def __next__(self):
    try:
        return next(self.it)
    except Exception as e:
        self.it = iter(self.loader)
        return next(self.it)