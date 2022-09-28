#Version 2.1
import sys
# sys.path.append('')
from model import Recognizer_CNN
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import numpy as np
import pdb
import copy
from model import SuCNN
import math
from model.model_structure import Model_Structure
from model import model_utils

celossnone = nn.CrossEntropyLoss(reduction='none')
celosssum = nn.CrossEntropyLoss(reduction='sum')
celossmean = nn.CrossEntropyLoss(reduction='mean')

class Train():
    def __init__(self, opt):
        self.modelMain = {}
        self.optimizer = {}
        self.modelSu = {}
        self.optimizerSu = {}

        self.opt = opt
        self.cl_dataset = opt.cl_dataset

        self.Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor
        self.TensorB = torch.cuda.ByteTensor if opt.cuda else torch.ByteTensor
        self.TensorL = torch.cuda.LongTensor if opt.cuda else torch.LongTensor
        self.model_maintain = Model_Structure()
        self.modelMaintainMode = None
        self.global_index = 0
        self.training_iterator = {}
        self.training_task_iterator = {}
        for label_index in range(opt.num_class):
            self.training_iterator[label_index] = self.cl_dataset.get_training_iterator(label_index)
            self.modelMain[label_index] = Recognizer_CNN(self.opt, label_index)
            self.optimizer[label_index] = torch.optim.SGD(
                    self.modelMain[label_index].parameters(), lr=0.1, momentum=0.9)

            if opt.cuda:
                self.modelMain[label_index].cuda()
    def initial_task(self):
        self.cl_dataset.intialization(TaskorClass=True)

        for label_index in range(self.opt.num_tasks):
            self.training_task_iterator[label_index] = self.cl_dataset.get_training_iterator(label_index)
            self.modelSu[label_index] = SuCNN(self.opt, label_index)
            self.optimizerSu[label_index] = torch.optim.SGD(
                    self.modelSu[label_index].parameters(), lr=0.1, momentum=0.9)

            if self.opt.cuda:
                self.modelSu[label_index].cuda()
	
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        BATCH_SIZE = real_data.shape[0]

        if self.opt.dataset == 'mnist':
            alpha = torch.rand(BATCH_SIZE, 1)
        else:
            alpha = torch.rand(BATCH_SIZE, 1)

        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates, _, _ = netD(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(
                                    disc_interpolates.size()).cuda(),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1))**8).mean()
        return gradient_penalty

    def parameter_l2_penalty(self, models):
        
        weight = (models.weight1 - self.model_maintain.weight1.data).norm()
        weight = weight + (models.weight2 - self.model_maintain.weight2.data).norm()
        weight = weight + (models.weight3 - self.model_maintain.weight3.data).norm()
        
        return weight

    def learn_sharedmodel(self):
        weights_list = []
        for label_index in range(self.global_index + 1):
            weights_list.append(self.modelMain[label_index].get_parameters())
        self.model_maintain.share_mean(weights_list)

    def set_model_parameters(self):
        self.modelMain[self.global_index].set_parameters(self.model_maintain)


    def train_classes(self, label_index_, i_batch):
        
        finished_epoch = 0
        loss_pen_n = 0
        constraints = 0
        mainloss_p = 0
        loss_pen_pn = 0
        imgs = self.training_iterator[label_index_].__next__()

        if imgs is None:
            return 0, -1, 0, 0, 0, 0

        finished_epoch = self.training_iterator[label_index_]._finished_epoch
        self.modelMain[label_index_].set_train_info(i_batch, finished_epoch)
        positive_data = self.Tensor(imgs[0])

        ########## Gradient penalty
        loss_pen = self.calc_gradient_penalty(self.modelMain[label_index_], positive_data, positive_data)
        ########## Reconstruction Loss
        score_temp_0, z, hlist = self.modelMain[label_index_].forward(positive_data)
        ########## Constraints
        for para in self.modelMain[label_index_].parameters():
            para_temp_norm = para.norm(2, dim=1)
            constraints =  constraints + ((para_temp_norm - 0.0)**8).mean() # (para.norm()).mean()#

        ########### MainLoss
        mainloss_p = mainloss_p + torch.log(torch.sigmoid(1.0 * score_temp_0) + 1e-2).mean()
        ################################################################################

        loss = - 1.0 * mainloss_p + 0.5 * loss_pen

        for temp_class_index in range(label_index_ + 1):
            self.optimizer[temp_class_index].zero_grad()

        loss.backward()
        #update current tasks
        torch.nn.utils.clip_grad_norm_(self.modelMain[label_index_].parameters(), self.opt.gradient_clip)
        self.optimizer[label_index_].step()

        return loss_pen.data, finished_epoch, constraints.data, loss_pen_n, loss_pen_pn, mainloss_p.data


    def train_tasks(self, label_index_):
        
        finished_epoch = 0

        imgs = self.training_task_iterator[label_index_].__next__()

        if imgs is None:
            return 0, -1
        finished_epoch = self.training_task_iterator[label_index_]._finished_epoch
        positive_data = self.Tensor(imgs[0])
        target = self.TensorL(imgs[1]) - self.opt.task_bounds[label_index_]
        #Reconstruction Loss
        scores_su = self.modelSu[label_index_].forward(positive_data)
        loss = celossmean(scores_su, target)
        self.optimizerSu[label_index_].zero_grad()
        loss.backward()

	    #update current tasks
        torch.nn.utils.clip_grad_norm_(self.modelSu[label_index_].parameters(), self.opt.gradient_clip)
        self.optimizerSu[label_index_].step()


        return loss.data, finished_epoch
    
    def learn_sharedmodel_w(self, temp_index):

        weights_list = []
        for label_index in range(self.global_index + 1):
            if temp_index == label_index:
                continue
            weights_list.append(self.modelMain[label_index].get_parameters())
            # pdb.set_trace()
        if len(weights_list) != 0:
            self.modelMain[temp_index].share_model.share_mean(weights_list)

    def eval(self):
        self.learn_sharedmodel()
        model_utils.calculateFisherMatrix(self)
        model_utils.UpdateMultiTaskWeightWithAlphas(self.global_index, self.modelMain)
        self.modelMaintainMode = Model_Structure()
        model_utils.CalculateModeParametes(self)
        model_utils.CalculateModePerClassParametes(self)
        # pdb.set_trace()
        for label_test in range(self.opt.num_class):
            self.learn_sharedmodel_w(label_test)
            self.modelMain[label_test].eval()
            self.modelMain[label_test].eval_flag = True

    def train(self):
        for label_test in range(self.opt.num_class):
            self.modelMain[label_test].train()
            self.modelMain[label_test].eval_flag = False

    def forward(self, index, imgs):
        return self.modelMain[index].forward_eval_21(imgs, self.model_maintain)

    def forwardFinal(self, index, imgs):

        score_cl = []
        for temp_class_index in range(self.opt.task_bounds[index], self.opt.task_bounds[index + 1]):
            score_cl.append(self.modelMain[temp_class_index].forward_eval_21(imgs, self.model_maintain))
        
        score_cl = torch.cat(score_cl, dim=1)
        score_su =  self.modelSu[index].forward(imgs)
        score_su = torch.softmax(score_su, dim=1)
        k=1
        max_value = score_su.topk(k)[0][:, k-1]
        mask1 = (score_su >= max_value.unsqueeze(1)).float()

        return  torch.sigmoid(score_cl) * mask1


    def saveModel(self):
        torch.save(self.modelMain[self.global_index].state_dict(), '../savedModel/' + str(self.opt.gpu) + str(self.global_index) + '.pkl')
        print('Model saved!')

    def loadModel(self):
        self.modelMain[self.global_index].load_state_dict(torch.load('../savedModel/' + str(self.opt.gpu) + str(self.global_index) + '.pkl'))
        print('Model loaded!')





    