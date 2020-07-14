import numpy as np
import pandas as pd

import torch.optim as optim
import torch.nn as nn
import os
import torch
import random

from network import net

import matplotlib.pyplot as plt


def calculateStockPrice(ori, diffseq):
    output = []
    for i in range(len(diffseq)):
        output.append(ori+diffseq[i])
        ori = ori + diffseq[i]
    return output


# 读取数据

file = "data.csv"
data = np.loadtxt(file, delimiter=",")

trainset = data[0:-161]
testset = data[-161:]

# 制造测试数据

lastprice = testset[150, 0]
test_input = testset[0:150, :]
tmp1 = testset[150:160, 0]
tmp2 = testset[151:, 0]

test_output = tmp2 - tmp1
test_output = test_output * 10


# 因为要输入到模型进行预测，所以要调整尺寸
test_input = torch.from_numpy(test_input)
test_input = torch.unsqueeze(test_input, 0)
test_input = torch.unsqueeze(test_input, 0)
test_input = test_input.float()

trainnum = trainset.shape[0]
maxindex = trainnum - 160


# 从原始数据中抽取按时间段，构成一个个训练集的输入输出

for i in range(50):
    if i == 0:
        randindex = random.randint(0, maxindex)
        input = trainset[randindex:randindex + 150]

        # 这里选择两日之间的股价差值作为output
        tmp1 = trainset[randindex+150:randindex+160, 0]
        tmp2 = trainset[randindex+151:randindex+160, 0]
        tmp2 = np.append(tmp2, trainset[160, 0])
        output = tmp2 - tmp1

        input_tensor = input
        output_tensor = output

    else:
        randindex = random.randint(0, maxindex)
        input = trainset[randindex:randindex+150]

        tmp1 = trainset[randindex+150:randindex+160, 0]
        tmp2 = trainset[randindex+151:randindex+160, 0]
        tmp2 = np.append(tmp2, trainset[160, 0])
        output = tmp2 - tmp1

        input_tensor = np.dstack((input_tensor, input))
        output_tensor = np.dstack((output_tensor, output))


train_input = torch.from_numpy(input_tensor)
train_output = torch.from_numpy(output_tensor)
train_output = train_output * 10

# 构建模型

model = net()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

if os.path.exists('checkpoint/model.pkl'):
    print('load model')
    model.load_state_dict(torch.load('checkpoint/model.pkl'))

for epoch in range(100):
    train_acc = 0
    train_loss = 0
    min_loss = 1e5

    randindex = random.randint(0, 49)

    optimizer.zero_grad()

    input = torch.randn(1, 1, 32, 32)

    # preds = model(input)

    modelinput = train_input[:, :, randindex]
    modelinput = torch.unsqueeze(modelinput, 0)
    modelinput = torch.unsqueeze(modelinput, 0)
    modelinput = modelinput.float()

    preds = model(modelinput)

    realoutput = train_output[0, :, randindex]
    realoutput = torch.unsqueeze(realoutput, 0)
    realoutput = realoutput.float()
    realoutput = realoutput * 10

    train_loss = loss_fn(preds, realoutput)
    train_loss += train_loss.item()
    print('train loss: ', train_loss)

    train_loss.backward()
    optimizer.step()

    plt.figure()

    if epoch % 1 == 0:

        # 预测的时候要把差值加上股价原值
        test_preds = model(test_input)

        # print("test_preds: ", test_preds)

        x = np.linspace(1, 10, 10)

        y1 = test_preds.detach().numpy()
        y2 = test_output
        y1 = np.squeeze(y1, 0)

        y1 = calculateStockPrice(lastprice, y1)
        y2 = calculateStockPrice(lastprice, y2)

        plt.xlim((0, 11))
        plt.ylim((-50, 100))

        plt.plot(x, y1, label="predict curve")
        plt.plot(x, y2, label="true curve")
        plt.show()

    if epoch % 20 == 0:
        print('save model')
        torch.save(model.state_dict(), 'checkpoint/model.pkl')
