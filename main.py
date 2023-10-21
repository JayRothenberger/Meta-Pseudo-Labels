import torch
import torchvision
from torchvision.transforms import v2
from util import accuracy, RepeatLoader
from mpl import MPL
torch.manual_seed(42)
transform = v2.Compose(
                    [
                        torchvision.transforms.Resize(size=(28, 28), antialias=True),
                        torchvision.transforms.ToTensor(),
                        torch.flatten
                    ]
                    )
train = torchvision.datasets.MNIST('/scratch/jroth/mnist/', True, download=True,transform =transform)
val = torchvision.datasets.MNIST('/scratch/jroth/mnist/', False, download=True, transform =transform)

train_kwargs = {'batch_size': 256}

cuda_kwargs = {'num_workers': 12,
                'pin_memory': True,
                'shuffle': False}

train_kwargs.update(cuda_kwargs)

train_loader = torch.utils.data.DataLoader(train, **train_kwargs)
val_loader = torch.utils.data.DataLoader(val, **train_kwargs)
# data loaders that repeat forever
train = RepeatLoader(train_loader)
val = RepeatLoader(val_loader)
# define the two models
student = torchvision.ops.MLP(784, [512, 256, 10], torch.nn.BatchNorm1d, torch.nn.ReLU, True, 0.1)
teacher = torchvision.ops.MLP(784, [512, 256, 10], torch.nn.BatchNorm1d, torch.nn.ReLU, True, 0.1)
# their optimizers
student_opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.1, weight_decay=1e-5)
teacher_opt = torch.optim.SGD(teacher.parameters(), lr=0.1, momentum=0.1, weight_decay=1e-5)
# train for 100 MPL weight updates, more is better but improvement beyond trivial accuracy is clearly observable in 100
for i, ((U, _), (L, y)) in enumerate(zip(train, val)):
    out = MPL(U, L, y, student, teacher, student_opt, teacher_opt)
    if i % 10 == 0:
        out_epoch = [accuracy(student(L), y)[0].item() for L, y in val_loader]
        print(f'epoch: {i // 10}, val accuracy:', sum(out_epoch) / len(out_epoch))
    if i > 100:
        break