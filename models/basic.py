import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.keep_prob1 = 0.2
        self.keep_prob2 = 0.5

        self.fn_loss = torch.nn.CrossEntropyLoss()  # 비용 함수에 소프트맥스 함수 포함되어져 있음.

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            # 1번째 conv layer : 입력 층 3, 출력 32, Relu, Poolling으로 MAX 직용.
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # 2번째 conv layer : 입력 층 32, 출력 64, Relu, Poolling으로 MAX 직용.
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # 3번째 conv layer : 입력 층 64, 출력 128, Relu, Polling으로 Max 적용.
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        self.fc1 = nn.Linear(self.args.flatten_dim_basic, 1250, bias=True)  # fully connected,
        nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU()
        )  # dropout 적용
        self.fc2 = nn.Linear(1250, self.args.num_classes, bias=True)  # 오류패턴 9개로 출력 9
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # fully conntect를위해 flatten을 함.
        out = self.layer4(out)
        out = self.fc2(out)
        return out


class CNNDeepAugmentBasic(nn.Module):
    def __init__(self, args):
        super(CNNDeepAugmentBasic, self).__init__()
        self.args = args

        self.fn_loss = None     # todo: implement this

        self.layer1 = nn.Sequential(
            nn.Conv2d(args.num_channel, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.6)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.6)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(args.flatten_dim_basic_deep_augment, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, args.num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.layer3(out)
        return out