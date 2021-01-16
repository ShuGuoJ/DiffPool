import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from networks import  Net
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import random_split
from visdom import Visdom
import numpy as np
from Trainer import Trainer
from utils import count_max_nodes
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=100000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')

args = parser.parse_args()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset)
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)
training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])



train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
max_nodes = count_max_nodes(dataset)
model = Net(dataset.num_features, 64, dataset.num_classes, max_nodes)
trainer = Trainer(model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss()



# def eval(model,loader):
#     model.eval()
#     correct = 0.
#     loss = 0.
#     for data in loader:
#         data = data.to(args.device)
#         out = model(data)
#         pred = out.max(dim=1)[1]
#         correct += pred.eq(data.y).sum().item()
#         loss += F.nll_loss(out,data.y,reduction='sum').item()
#     return correct / len(loader.dataset),loss / len(loader.dataset)


min_loss = 1e10
patience = 0
viz = Visdom(port=17000)
viz.line([[0., 0.]], [0], win='train&eval loss', opts={'title':'train&eval loss',
                                                       'legend':['train_loss', 'eval_loss']})
viz.line([0.], [0], win='accuracy', opts={'title':'accuracy',
                                          'legend':['accuracy']})
for epoch in range(args.epochs):
    # model.train()
    # losses = list()
    # for i, data in enumerate(train_loader):
    #     data = data.to(args.device)
    #     out = model(data)
    #     loss = F.nll_loss(out, data.y)
    #     print("Training loss:{}".format(loss.item()))
    #     losses.append(loss.item())
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    # val_acc,val_loss = eval(model,val_loader)
    train_loss = trainer.train(train_loader, optimizer, criterion, DEVICE)
    val_loss, acc = trainer.evaluate(val_loader, criterion, DEVICE)
    viz.line([[train_loss, val_loss]], [epoch], win='train&eval loss', update='append')
    viz.line([acc], [epoch], win='accuracy', update='append')
    print("Validation loss:{}\taccuracy:{}".format(val_loss, acc))
    if val_loss < min_loss:
        torch.save(model.state_dict(),'latest.pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break

model = Net(dataset.num_features, 64, dataset.num_classes, max_nodes)
model.load_state_dict(torch.load('latest.pth'))
trainer.model = model
test_acc,test_loss = trainer.evaluate(test_loader, criterion, DEVICE)
print("Test accuarcy:{}".format(test_acc))
