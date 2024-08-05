import argparse

class Options():
    def initialize(self):

        parser = argparse.ArgumentParser(description='PyTorch ResNet18 Example')
        #parser.add_argument('--outf', './model/',  help='folder to output images and model checkpoints.')
        #parser.add_argument('--net', './model/Resnet18.pth', help='path to net (to continue training)')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for SGD')
        parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
        parser.add_argument('--train_batch_size', type=int, default=10, metavar='N', help = 'input batch size for training (default: 128)')
        parser.add_argument('--test_batch_size', type=int, default=10, metavar='N', help = 'input batch size for testing (default: 100)')
        parser.add_argument('--Epoch', type=int, default=20, help='the starting epoch count')
        parser.add_argument('--no_train', action='store_true', default=False, help = 'If train the Model')
        parser.add_argument('--save_model', action='store_true', default=False, help = 'For Saving the current Model')
        parser.add_argument('--model', type=str, default='resnet18', help='The model to be trained')
        parser.add_argument('--img_size', type=int, default=224, help='img_size')

        args = parser.parse_args()

        return args
