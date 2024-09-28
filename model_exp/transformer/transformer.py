#!/usr/bin/env python3
import time
import os
import shutil
import argparse

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.utils import save_image
import sys

sys.path.append('../')

import extension as ext
from model import *


class MNIST:
    def __init__(self):
        self.cfg = self.add_arguments()
        self.model_name = self.cfg.dataset + '_' + self.cfg.arch + '_d' + str(self.cfg.depth) + '_w' + str(self.cfg.width)+ '_' + ext.normalization.setting(self.cfg)
        self.model_name = self.model_name + '_lr' + str(self.cfg.lr) + '_bs' + str(
            self.cfg.batch_size[0]) + '_seed' + str(self.cfg.seed)
        print(self.cfg.norm_cfg)
        self.result_path = os.path.join(self.cfg.output, self.model_name, self.cfg.log_suffix)
        os.makedirs(self.result_path, exist_ok=True)
        self.logger = ext.logger.setting('log.txt', self.result_path, self.cfg.test, bool(self.cfg.resume))

        self.logger = ext.logger.setting('log.txt', self.result_path, self.cfg.test, self.cfg.resume is not None)
        ext.trainer.setting(self.cfg)
        if self.cfg.arch =='resnet18':
            self.model = resnet18(num_classes=10)
        elif self.cfg.arch =='resnet34':
            self.model = resnet34(num_classes=10)
        elif self.cfg.arch =='resnet50':
            self.model = resnet50(num_classes=10)
        else:
            self.model = MLP(width=self.cfg.width, depth=self.cfg.depth)
        self.logger('==> model [{}]: {}'.format(self.model_name, self.model))
        self.optimizer = ext.optimizer.setting(self.model, self.cfg)
        self.scheduler = ext.scheduler.setting(self.optimizer, self.cfg)

        self.saver = ext.checkpoint.Checkpoint(self.model, self.cfg, self.optimizer, self.scheduler, self.result_path,
                                               not self.cfg.test)
        self.saver.load(self.cfg.load)

        # dataset loader
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        self.train_loader = ext.dataset.get_dataset_loader(self.cfg, transform, train=True, use_cuda=False)
        self.val_loader = ext.dataset.get_dataset_loader(self.cfg, transform, train=False, use_cuda=False)

        self.device = torch.device('cuda')
        # self.device = torch.device('cpu')
        self.num_gpu = torch.cuda.device_count()
        self.logger('==> use {:d} GPUs'.format(self.num_gpu))
        if self.num_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()

        self.best_acc = 0
        if self.cfg.resume:
            saved = self.saver.resume(self.cfg.resume)
            self.cfg.start_epoch = saved['epoch']
            self.best_acc = saved['best_acc']
        self.criterion = nn.MSELoss() if self.cfg.arch == 'AE' else nn.CrossEntropyLoss()

        self.vis = ext.visualization.setting(self.cfg, self.model_name,
                                             {'train loss': 'loss', 'test loss': 'loss', 'train accuracy': 'accuracy',
                                              'test accuracy': 'accuracy'})
        return

    def add_arguments(self):
        parser = argparse.ArgumentParser('MNIST Classification')
        model_names = ['resnet18', 'resnet34', 'resnet50']
        parser.add_argument('-a', '--arch', metavar='ARCH', default=model_names[0], choices=model_names,
                            help='model architecture: ' + ' | '.join(model_names))
        parser.add_argument('-width', '--width', type=int, default=100)
        parser.add_argument('-depth', '--depth', type=int, default=4)
        ext.trainer.add_arguments(parser)
        parser.set_defaults(epochs=10)
        ext.dataset.add_arguments(parser)
        parser.set_defaults(dataset='cifar10', workers=1, batch_size=[64, 1000])
        ext.scheduler.add_arguments(parser)
        parser.set_defaults(lr_method='fix', lr=1e-3)
        ext.optimizer.add_arguments(parser)
        parser.set_defaults(optimizer='adam', weight_decay=1e-5)
        ext.logger.add_arguments(parser)
        ext.checkpoint.add_arguments(parser)
        ext.normalization.add_arguments(parser)
        ext.visualization.add_arguments(parser)
        args = parser.parse_args()
        if args.resume:
            args = parser.parse_args(namespace=ext.checkpoint.Checkpoint.load_config(args.resume))
        return args

    def train(self):
        # print("trainbegin:")
        if self.cfg.test:
            self.validate()
            return
        # train model
        for epoch in range(self.cfg.start_epoch + 1, self.cfg.epochs):
            if self.cfg.lr_method != 'auto':
                self.scheduler.step()
            self.train_epoch(epoch)
            accuracy, val_loss = self.validate(epoch)
            self.saver.save_checkpoint(epoch=epoch, best_acc=self.best_acc)
            if self.cfg.lr_method == 'auto':
                self.scheduler.step(val_loss)
        # finish train
        now_date = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
        self.logger('==> end time: {}'.format(now_date))
        new_log_filename = '{}_{}_{:5.2f}%.txt'.format(self.model_name, now_date, self.best_acc)
        self.logger('\n==> Network training completed. Copy log file to {}'.format(new_log_filename))
        shutil.copy(self.logger.filename, os.path.join(self.result_path, new_log_filename))
        # print("trainend.")
        return

    def train_epoch(self, epoch):
        self.logger('\nEpoch: {}, lr: {:.2g}, weight decay: {:.2g} on model {}'.format(epoch,
            self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[0]['weight_decay'], self.model_name))
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        progress_bar = ext.ProgressBar(len(self.train_loader))
        for i, (inputs, targets) in enumerate(self.train_loader, 1):
            inputs = inputs.to(self.device)
            targets = inputs if self.cfg.arch == 'AE' else targets.to(self.device)
            #print(inputs.size())
            # compute output
            outputs = self.model(inputs)
            losses = self.criterion(outputs, targets)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            # measure accuracy and record loss
            train_loss += losses.item() * targets.size(0)
            if self.cfg.arch == 'AE':
                correct = -train_loss
            else:
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(targets.view_as(pred)).sum().item()
            total += targets.size(0)
            if i % 10 == 0 or i == len(self.train_loader):
                progress_bar.step('Loss: {:.5g} | Accuracy: {:.2f}%'.format(train_loss / total, 100. * correct / total),
                    10)
        train_loss /= total
        accuracy = 100. * correct / total
        self.vis.add_value('train loss', train_loss)
        self.vis.add_value('train accuracy', accuracy)
        self.logger(
            'Train on epoch {}: average loss={:.5g}, accuracy={:.2f}% ({}/{}), time: {}'.format(epoch, train_loss,
                accuracy, correct, total, progress_bar.time_used()))
        return

    def validate(self, epoch=-1):
        test_loss = 0
        correct = 0
        total = 0
        progress_bar = ext.ProgressBar(len(self.val_loader))
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = inputs if self.cfg.arch == 'AE' else targets.to(self.device)
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, targets).item() * targets.size(0)
                if self.cfg.arch == 'AE':
                    correct = -test_loss
                else:
                    prediction = outputs.max(1, keepdim=True)[1]
                    correct += prediction.eq(targets.view_as(prediction)).sum().item()
                total += targets.size(0)
                progress_bar.step('Loss: {:.5g} | Accuracy: {:.2f}%'.format(test_loss / total, 100. * correct / total))
        test_loss /= total
        accuracy = correct * 100. / total
        self.vis.add_value('test loss', test_loss)
        self.vis.add_value('test accuracy', accuracy)
        self.logger('Test on epoch {}: average loss={:.5g}, accuracy={:.2f}% ({}/{}), time: {}'.format(epoch, test_loss,
            accuracy, correct, total, progress_bar.time_used()))
        if not self.cfg.test and accuracy > self.best_acc:
            self.best_acc = accuracy
            self.saver.save_model('best.pth')
            self.logger('==> best accuracy: {:.2f}%'.format(self.best_acc))
        if self.cfg.arch == 'AE':
            pic = to_img(outputs[:64].cpu().data)
            save_image(pic, os.path.join(self.result_path, 'result_{}.png').format(epoch))
        return accuracy, test_loss


if __name__ == '__main__':
    Cs = MNIST()
    torch.set_num_threads(1)
    Cs.train()
