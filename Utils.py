# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 16:24:47 2020

@author: Matt Sehgal
"""

import os
import numpy as np
from numpy import empty
import errno
import torchvision
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import transforms
    
    
# Data Loader
class Loader:
    ''' Loads and shapes data '''
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
    
    def get_loader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset, self.batch_size, shuffle=True)
    
    def get_dataset(self):
        return torchvision.datasets.ImageFolder(
            root=self.path,
            transform=transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            ]))
    
# Model Trainer
class Trainer:
    '''Initializes model weights, prepares data sets for training, trains models '''
    def __init__(self, loss, device):
        self.loss = loss
        self.device = device
        
    def init_weights(m):
        name = m.__class__.__name__
        if name.find('Conv') != -1 or name.find('BatchNorm') != -1:
            m.weight.data.normal_(0.00, 0.02)
    
    # Generate random noise vector Z
    def gen_noise(self, size):
        z = Variable(torch.randn(size, 100, device=self.device))
        if torch.cuda.is_available(): 
            return z.cuda()
        return z
    
    # Create data targets
    def real_data_target(self, size):
        rdt = Variable(torch.ones(size, 1, device=self.device))
        if torch.cuda.is_available(): return rdt.cuda()
        return rdt
    
    def fake_data_target(self, size):
        fdt = Variable(torch.zeros(size, 1, device=self.device))
        if torch.cuda.is_available(): return fdt.cuda()
        return fdt
    
    # Create optimizers
    def create_optimizers(self, d_parameters, g_parameters):
        self.d_opt = Adam(d_parameters, lr=0.0003, betas=(0.5, 0.999))
        self.g_opt = Adam(g_parameters, lr=0.0002, betas=(0.5, 0.999))
    
    def train_discriminator(self, discriminator, real_data, fake_data):
        # Reset gradients
        self.d_opt.zero_grad()
        
        # 1.1 Train on Real Data
        prediction_real = discriminator(real_data)
        # Calculate error and backpropagate
        error_real = self.loss(prediction_real, self.real_data_target(real_data.size(0)))
        error_real.backward()
    
        # 1.2 Train on Fake Data
        prediction_fake = discriminator(fake_data)
        # Calculate error and backpropagate
        error_fake = self.loss(prediction_fake, self.fake_data_target(real_data.size(0)))
        error_fake.backward()
        
        # 1.3 Update weights with gradients
        self.d_opt.step()
        
        # Return error
        return error_real + error_fake, prediction_real, prediction_fake

    def train_generator(self, generator, discriminator, fake_data):
        # 2. Train Generator
        # Reset gradients
        self.g_opt.zero_grad()
        # Sample noise and generate fake data
        prediction = discriminator(fake_data)
        # Calculate error and backpropagate
        error = self.loss(prediction, self.real_data_target(prediction.size(0)))
        error.backward()
        # Update weights with gradients
        self.g_opt.step()
        # Return error
        return error
    
# Data, Event, and Model Logger
class Logger:
    ''' Logs data, events, and model states'''
    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment)

    def log(self, d_error, g_error, epoch, n_batch, num_batches):

        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/D_error'.format(self.comment), d_error, step)
        self.writer.add_scalar(
            '{}/G_error'.format(self.comment), g_error, step)

    def log_images(self, images, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):
        '''
        input images are expected in format (NCHW)
        '''
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)
        
        if format=='NHWC':
            images = images.transpose(1,3)
        

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.comment, '')

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=normalize, scale_each=True)
        # Make vertical grid from image tensor
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(
            images, nrow=nrows, normalize=True, scale_each=True)

        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, horizontal_grid, step)

        # Save plots
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)

        # Plot and save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch, 'hori')
        plt.close()

        # Save squared
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, comment=''):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir,
                                                         comment, epoch, n_batch))

    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):
        
        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
        if isinstance(d_pred_real, torch.autograd.Variable):
            d_pred_real = d_pred_real.data
        if isinstance(d_pred_fake, torch.autograd.Variable):
            d_pred_fake = d_pred_fake.data
        
        
        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch,num_epochs, n_batch, num_batches)
             )
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

    def save_models(self, generator, discriminator, epoch):
        out_dir = './data/models/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(),
                   '{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),
                   '{}/D_epoch_{}'.format(out_dir, epoch))

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                
class Plotter:
    '''Plots epoch and batch results'''
    def __init__(self, num_epochs, num_batches):
        #epoch plotting data
        self.epoch = np.arange(0, num_epochs, 1)
        self.d_error = empty(num_epochs)
        self.g_error = empty(num_epochs)
        self.epoch_dx = empty(num_epochs)
        self.epoch_dgz = empty(num_epochs)
        #batch plotting data
        self.batch = np.arange(0, num_batches, 1)
        self.batch_dx = empty(num_batches)
        self.batch_dgz = empty(num_batches)
    
    def update_errors(self, epoch, d_error, g_error):
        self.d_error[epoch] = d_error
        self.g_error[epoch] = g_error
    
    def update_epoch_dx_dgz(self, epoch, d_pred_r, d_pred_f):
        self.epoch_dx[epoch] = d_pred_r
        self.epoch_dgz[epoch] = d_pred_f
    
    def update_batch_dx_dgz(self,batch, d_pred_r, d_pred_f):
        self.batch_dx[batch] = d_pred_r
        self.batch_dgz[batch] = d_pred_f
    
    def plot(self):
        plt.figure(1)
        plt.subplot(211)
        plt.plot(self.epoch, self.d_error, color='blue', marker='.')
        plt.ylabel("D Error")
        plt.subplot(212)
        plt.plot(self.epoch, self.g_error, color='green', marker='.')
        plt.xlabel("Epoch")
        plt.ylabel("G Error")
        plt.figure(2)
        plt.plot(self.epoch, self.epoch_dx, color='blue', label="P(D(x))", marker='.')
        plt.plot(self.epoch, self.epoch_dgz, color='green', label="P(D(G(z)))", marker='.')
        plt.xlabel("Epoch")
        plt.ylim(0,1)
        plt.legend()
        plt.figure(3)
        plt.plot(self.batch, self.batch_dx, color='blue', label="P(D(x))", marker='.')
        plt.plot(self.batch, self.batch_dgz, color='green', label="P(D(G(z)))", marker='.')
        plt.xlabel("Batch num")
        plt.legend()
        return plt
        
        
