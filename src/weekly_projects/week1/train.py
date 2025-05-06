import torch.optim
from weekly_projects.models.ffn import FFN
from weekly_projects.data_loaders import mnist_loader

from torch import nn
from weekly_projects.week1.trainer import train

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FFN(args.hidden).to(device)
    if args.datadir is not None:
        train_dtl, test_dtl = mnist_loader.get_loaders(batch_size=args.batch, root=args.datadir)
    else:
        train_dtl, test_dtl = mnist_loader.get_loaders(batch_size=args.batch)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr)

    if torch.cuda.is_available():
        model.cuda()

    train(model=model,
          train_dtl=train_dtl,
          test_dtl=test_dtl,
          criterion=criterion,
          optimizer=optimizer,
          epochs=args.epochs,
          device=device)
