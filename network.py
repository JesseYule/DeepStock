import torch
import torch.nn as nn


class net(nn.Module):

    def __init__(self):
        super(net, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, 5)
        self.conv2 = nn.Conv2d(1, 1, 5)
        self.conv3 = nn.Conv2d(1, 1, 5)
        self.conv4 = nn.Conv2d(1, 1, 5)
        self.conv5 = nn.Conv2d(1, 1, 5)
        self.conv6 = nn.Conv2d(1, 1, 5)
        self.conv7 = nn.Conv2d(1, 1, 5)
        self.conv8 = nn.Conv2d(1, 1, 5)
        self.conv9 = nn.Conv2d(1, 1, 5)
        self.conv10 = nn.Conv2d(1, 1, 5)
        self.conv11 = nn.Conv2d(1, 1, 5)
        self.conv12 = nn.Conv2d(1, 1, 5)

        self.fc1 = nn.Linear(18544, 4000)
        self.fc2 = nn.Linear(4000, 1000)
        self.fc3 = nn.Linear(1000, 10)
        self.fc4 = nn.Linear(5000, 1000)
        self.fc5 = nn.Linear(1000, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = torch.relu(x)

        x = self.conv3(x)
        x = torch.relu(x)

        x = self.conv4(x)
        x = torch.relu(x)

        x = self.conv5(x)
        x = torch.relu(x)

        x = self.conv6(x)
        x = torch.relu(x)

        x = self.conv7(x)
        x = torch.relu(x)

        # x = self.conv8(x)
        # x = torch.relu(x)

        # x = self.conv9(x)
        # x = torch.relu(x)
        #
        # x = self.conv10(x)
        # x = torch.tanh(x)
        #
        # x = self.conv11(x)
        # x = torch.tanh(x)
        #
        # x = self.conv12(x)
        # x = torch.tanh(x)

        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        x = torch.tanh(x)

        x = self.fc2(x)
        x = torch.tanh(x)

        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
