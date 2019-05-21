# this is the toy experiment for Virtual Adversarial Training.

import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors  import ListedColormap
import pandas as pd
import sklearn, torch
import torch.nn as nn
from sklearn import datasets
from utils import vat_loss
import torch.nn.functional as F
from torch.autograd import Variable

def initial_plot(y):
    markers = ('s','x','o','^')
    colors = ('red','blue','lightgreen','gray')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    return markers,colors,cmap

def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)

def kl_div_with_logit(q_logit, p_logit):

    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1).mean(dim=0)
    qlogp = ( q *logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp

def _l2_normalize(d):

    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2)) + 1e-16)
    return torch.from_numpy(d)

def vat_loss(model, ul_x, ul_y, xi=1e-3, eps=2.5, num_iters=1):

    # find r_adv

    d = torch.Tensor(ul_x.size()).normal_() # 所有元素的std =1, average = 0
    for i in range(num_iters):
        d = xi *_l2_normalize(d)
        d = Variable(d.cuda(), requires_grad=True)
        y_hat = model(ul_x + d)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()

        d = d.grad.data.clone().cpu()
        model.zero_grad()

    d = _l2_normalize(d)
    d = Variable(d.cuda())
    r_adv = eps *d
    # compute lds
    y_hat = model(ul_x + r_adv.detach())
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    return delta_kl

def train(model, x, y, ul_x, optimizer,opt):
# x is the labeled data with y being the target. ul_x is the unlabeled data.
    model.train()
    ce = nn.CrossEntropyLoss()
    y_pred = model(x)
    ce_loss = ce(y_pred, y)

    ul_y = model(ul_x)
    v_loss = vat_loss(model, ul_x, ul_y, eps=2.5)
    loss = v_loss + ce_loss
    loss = v_loss
    if opt.method == 'vatent':
        loss += entropy_loss(ul_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return v_loss, ce_loss
    # return  1, ce_loss

def val(model, x_validate,y_target):
    model.eval()
    y_pred = model(x_validate)
    y_prob = F.softmax(y_pred,dim=1)
    y_pred = y_pred.max(1)[1]
    acc = (y_pred==y_target).cpu().data.numpy().mean()
    return acc, y_prob[:,1]

def visualization_decision_boundary(model, x_labeled, y_labeled, x_unlabeled,y_unlabeled):
# def plot_decision_regions(X, y, net, resolution, cm):
    # cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#00FF00','#0000FF',])
    X = np.concatenate([x_labeled,x_unlabeled])
    resolution = 0.01
    X_numpy = x_labeled
    y_numpy = y_labeled
    markers, colors, cm = initial_plot(y_numpy)

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution)
                           )
    X_mesh = Variable(torch.FloatTensor(np.array([xx1.ravel(), xx2.ravel()]).T).cuda(), requires_grad=False)
    predicts = F.softmax(model(X_mesh),dim=1).cpu().data.numpy()[:,1]
    predicts = predicts.reshape(xx1.shape)

    plt.figure(2)
    plt.clf()
    plt.contourf(xx1, xx2, predicts, cmap=plt.cm.RdBu, alpha=.1)
    plt.colorbar()

    plt.scatter(unlabeled_data_train[:, 0], unlabeled_data_train[:, 1], c=y_unlabeled, cmap=cm_bright,alpha=0.2)
    plt.scatter(x_labeled[:, 0], x_labeled[:, 1], cmap=cm_bright, c=y_labeled, alpha=0.5)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.pause(0.0001)


    ul_y = model(torch.FloatTensor(x_unlabeled).float().cuda())
    v_loss = vat_loss(model, torch.FloatTensor(x_unlabeled).float().cuda(), ul_y, eps=2.5)
    # pass

class Net (nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(2,100),
            nn.ReLU(True),
            nn.Linear(100,50),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(50,2)
        )
    def forward(self, input):
        output = self.main(input)
        return output

class config():
    method = 'vatent'
    # method = None

opt = config()

np.random.seed(1)
torch.manual_seed(1)
dataset = datasets.make_moons(n_samples=2010,noise=0.12 ,shuffle=True,random_state=1)
(X, y) = dataset

labeled_number = 10
valid_number = 800
num_iter_per_epoch = 200
batch_size = 5
unlabeled_batch_size = 128
max_epochs = 102


labeled_data_train, labeled_target = X[:labeled_number], y[:labeled_number]
unlabeled_data_train,unlabeled_data_target = X[labeled_number:-valid_number],y[labeled_number:-valid_number]
valid_data, valid_target = X[-valid_number:],y[-valid_number:]

# visualize the dataset
color_class = np.zeros(y.shape)
color_class[labeled_number:-valid_number]=1
color_class[-valid_number:]=2

net = Net().cuda()
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(net.parameters(),lr =1e-3)

for epoch in range(max_epochs):
    for i in range(num_iter_per_epoch):
        batch_indices = torch.LongTensor(np.random.choice(labeled_data_train.shape[0], batch_size, replace=False))
        x = labeled_data_train[batch_indices]
        y = labeled_target[batch_indices]
        batch_indices_unlabeled = torch.LongTensor(
            np.random.choice(unlabeled_data_train.shape[0], unlabeled_batch_size, replace=False))
        ul_x = unlabeled_data_train[batch_indices_unlabeled]

        v_loss, ce_loss = train(net.train(), torch.FloatTensor(x).float().cuda(), torch.LongTensor(y).cuda(), torch.FloatTensor(ul_x).float().cuda(),
                                optimiser,opt=opt)
        if i % 10 == 0:
            visualization_decision_boundary(net,labeled_data_train,labeled_target,unlabeled_data_train,unlabeled_data_target)
            # print("Epoch :", epoch, "Iter :", i, "VAT Loss :", v_loss.item(), "CE Loss :", ce_loss.item())

    valid_acc = val(model=net, x_validate= torch.FloatTensor(valid_data).float().cuda(),y_target=torch.LongTensor(valid_target).cuda())
    print(valid_acc[0])
