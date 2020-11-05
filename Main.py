# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:45:50 2020

@author: Matt Sehgal
"""

from IPython import display
from Utils import Loader, Trainer, Plotter, Logger
from Networks import Discriminator, Generator
import torch
from torch import nn
from torch.autograd import Variable

# Load data -- hardcoded
data_path = "NEED PATH"
data_name = "NEED NAME"
batch_size = 250
loader = Loader(data_path, batch_size)
data_loader = loader.get_loader(loader.get_dataset())
num_batches = len(data_loader)

# Create newtork instances
discriminator = Discriminator()
discriminator.apply(Trainer.init_weights)
generator = Generator()
generator.apply(Trainer.init_weights)

# Create trainer and initialize network weights
device = "cpu"
if torch.cuda.is_available():
    print("cuda available")
    device = "cuda:0"
    discriminator.cuda()
    generator.cuda()

# Optimizers and loss function
net_trainer = Trainer(nn.BCELoss(), device)
net_trainer.create_optimizers(discriminator.parameters(),generator.parameters())

# Number of epochs -- hardcoded
num_epochs = 200

# Testing samples -- hardcoded
num_test_samples = 16
test_noise = net_trainer.gen_noise(num_test_samples)

# Start Training
logger = Logger(model_name='DCGAN', data_name=data_name)
plotter = Plotter(num_epochs, num_batches)

for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
        
        ### Train Discriminator
        real_data = Variable(real_batch)
        if torch.cuda.is_available(): real_data = real_data.cuda()
        # Generate fake data
        fake_data = generator(net_trainer.gen_noise(real_data.size(0))).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = net_trainer.train_discriminator(discriminator, 
                                                                            real_data, fake_data)

        ### Train Generator
        # Generate fake data
        fake_data = generator(net_trainer.gen_noise(real_batch.size(0)))
        # Train G
        g_error = net_trainer.train_generator(generator, discriminator, fake_data)
        
        # Log error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        
        # Batch plotting
        plotter.update_batch_dx_dgz(n_batch, d_pred_real.mean(), d_pred_fake.mean())
        
        # Display Progress
        data = data_loader
        batch_size = data_loader.batch_size
        
        if (n_batch) % (len(data)/batch_size) == 0:
            display.clear_output(True)
            # Display Images
            test_images = generator(test_noise).data.cpu()
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
            
            # Plotting
            plotter.update_errors(epoch, d_error, g_error)
            plotter.update_epoch_dx_dgz(epoch, d_pred_real.mean(), d_pred_fake.mean())
            
            # Plotting
            plotter.plot().show()
            
        # Model Checkpoints
        logger.save_models(generator, discriminator, epoch)
