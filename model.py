import torch.nn as nn

class VAT(nn.Module):

      def __init__(self, top_bn=True):

            super(VAT, self).__init__()
            self.top_bn = top_bn
            self.main = nn.Sequential(
                  nn.Conv2d(3, 128, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.1),

                  nn.MaxPool2d(2, 2, 1),
                  nn.Dropout2d(),

                  nn.Conv2d(128, 256, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(256, 256, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(256, 256, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1),

                  nn.MaxPool2d(2, 2, 1),
                  nn.Dropout2d(),

                  nn.Conv2d(256, 512, 3, 1, 0, bias=False),
                  nn.BatchNorm2d(512),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(512, 256, 1, 1, 1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(256, 128, 1, 1, 1, bias=False),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.1),

                  nn.AdaptiveAvgPool2d((1, 1))
                  )

            self.linear = nn.Linear(128, 10)
            self.bn = nn.BatchNorm1d(10)

      def forward(self, input):
            output = self.main(input)
            output = self.linear(output.view(input.size()[0], -1))
            if self.top_bn:
                  output = self.bn(output)
            return output