from test import make_test_step
import os
import numpy as np

import torch.nn as nn
import torch.optim
from utils import collaborative_sampling, ones_target, plot, zeros_target


def train(gen, discr, ae, loader, val, epochs, count, name, loss, optim_g, optim_d, device, gan, ae_lb, scheduler, collab, cgan):


    print("Training for " + str(epochs) +  " epochs, " + str(count) + " mini-batches per epoch")
    

    loss = nn.MSELoss() if loss == "L2" else nn.L1Loss()

    # create the functions used at each step of the training / testing process
    train_step = make_train_step(gen, discr, ae, loss, gan, ae_lb, cgan, optim_g, optim_d)
    test_step = make_test_step(gen, discr, ae, loss, gan, ae_lb, cgan, collab=False)

    cuda = torch.cuda.is_available()

    # Initialize all the buffers to save the losses
    losses, val_losses, losses_gan, val_losses_gan, losses_ae, val_losses_ae = [],[],[],[],[],[]
    loss_buffer, loss_buffer_gan, loss_buffer_ae = [],[],[]
    losses_normal, loss_normal_buffer = [],[]
    
    # Initialize our two scheduler, one for the generator and one for the discriminator
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_g, factor=0.5, threshold=1e-2, cooldown=0, verbose=True)
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_d, factor=0.5, threshold=1e-2, cooldown=0, verbose=True)
   
    # Do we already start the additional networks ? no by default
    start_others = False

    for epoch in range(1, epochs+1):

        # correct the count variable if needed (the argument)
        total = len(loader)
        if (total < count or count < 0): 
            count = total 
        
        curr_count = 0

        for x_batch, y_batch in loader:

            # Put the networks and the data on the correct devices
           
            gen.to(device)
            discr.to(device)
            ae.to(device)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Train using the current batch, and save the losses
            loss_, loss_normal, loss_gan, loss_ae = train_step(x_batch, y_batch, start_others)

            loss_buffer.append(loss_) # Composite loss
            loss_normal_buffer.append(loss_normal) # Generator Loss
            loss_buffer_gan.append(loss_gan) # Discriminator Loss
            loss_buffer_ae.append(loss_ae) # Autoencoder loss

            # Stop if count reached
            curr_count += 1
            if (curr_count >= count): break

            # If 10 batches done, compute the averages over the last 10 batches for some graphs
            if (len(loss_buffer) % 10 == 0):
                              
                # We consider that 10 batches is enough to wait before starting the other networks
                start_others = True

                # Get average train loss
                losses.append(sum(loss_buffer)/len(loss_buffer))
                losses_gan.append(sum(loss_buffer_gan)/len(loss_buffer_gan))
                losses_ae.append(sum(loss_buffer_ae)/len(loss_buffer_ae))
                losses_normal.append(sum(loss_normal_buffer)/len(loss_normal_buffer))


                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [D loss: %f] [AE loss: %f]"
                    % (epoch, epochs, curr_count, count, losses[-1], losses_gan[-1], losses_ae[-1]))
                loss_buffer, loss_normal_buffer, loss_buffer_gan, loss_buffer_ae = [],[],[],[]

                # Compute average validation loss
                val_loss_buffer = []
                val_loss_buffer_gan = []
                val_loss_buffer_ae = []

                # For each sample in the validation data
                for x_val, y_val in val:

                    gen.to(device)
                    discr.to(device)
                    ae.to(device)
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)

                    # COmpute and save the losses
                    loss_, loss_gan, loss_ae, _ = test_step(x_val, y_val)
                    val_loss_buffer.append(loss_)
                    val_loss_buffer_gan.append(loss_gan)
                    val_loss_buffer_ae.append(loss_ae)

                # Compute and store the average val losses
                val_losses.append(sum(val_loss_buffer)/len(val_loss_buffer))
                val_losses_gan.append(sum(val_loss_buffer_gan)/len(val_loss_buffer_gan))
                val_losses_ae.append(sum(val_loss_buffer_ae)/len(val_loss_buffer_ae))

                # Every 50 batches, plot and decrease lr if needed with the scheduler
                if (len(losses) % 5 == 0):
                    plot(losses, val_losses, losses_gan, val_losses_gan, losses_normal, losses_ae, val_losses_ae, name, gan, ae_lb)
                    if scheduler:
                        scheduler_g.step(val_losses[-1])
                        scheduler_d.step(val_losses_gan[-1])
      

        # Save the model for the epoch
        torch.save({
            'gen_state_dict': gen.state_dict(),
            'optim_g_state_dict': optim_g.state_dict(),
            'discr_state_dict': discr.state_dict(),
            'optim_d_state_dict': optim_d.state_dict(),
            }, "out/" + name + "/models/model_" + str(epoch) + ".tar")

        # Save the losses
        np.save('out/' + name + '/losses/loss_train.npy', np.array(losses))
        np.save('out/' + name + '/losses/loss_test.npy', np.array(val_losses))
        np.save('out/' + name + '/losses/loss_train_gan.npy', np.array(losses_gan))
        np.save('out/' + name + '/losses/loss_test_gan.npy', np.array(val_losses_gan))
        np.save('out/' + name + '/losses/loss_train_ae.npy', np.array(losses_ae))
        np.save('out/' + name + '/losses/loss_test_ae.npy', np.array(val_losses_ae))


    # If collaborative GAN is enabled, do discriminator shaping
    print("Model trained")
    if collab and gan: discriminator_shaping(gen, discr, loader, 50, 50, optim_d, device) 
    print("Discriminator shaping done")
    

    # Do the final update on the graph
    plot(losses, val_losses, losses_gan, val_losses_gan, losses_normal, losses_ae, val_losses_ae, name, gan, ae_lb)

def make_train_step(generator, discriminator, ae, loss, lambda_d, lambda_ae, cgan, optimizer_g, optimizer_d):

    def train_step(x, y, start_others):


        N = y.size(0)
        loss_gan = nn.BCELoss()

        #################
        # Train generator
        #################
        optimizer_g.zero_grad()
        generator.train()
        discriminator.eval()
        ae.eval()

        # Generate output  distriminator
        yhat = generator(x)

        # Compute the normal loss (L1 or L2)
        loss_g_normal = loss(yhat, y) 

        # Compute the adversarial loss (want want to generate realistic data)
        loss_g_adv = 0
        if lambda_d and start_others:
            prediction = discriminator(yhat) if not cgan else discriminator(yhat, x) # use the correct discriminator if CGAN enabled
            loss_g_adv = loss_gan(prediction, ones_target(N)) 

        # Compute the autoencoder loss
        loss_g_ae = 0
        if lambda_ae and start_others:
            # Compute the L1 or L2 loss over the latent tensors
            latent_y, _ = ae(y)
            latent_yhat, _ = ae(yhat)
            loss_g_ae = loss(latent_yhat, latent_y) 

        # Compute the composite loss
        loss_g = (loss_g_normal + lambda_d*loss_g_adv + lambda_ae*loss_g_ae)
        
        # Propagate the gradiants
        if start_others:
            loss_g.backward()
            optimizer_g.step()

        #####################
        # Train discriminator
        #####################
        loss_d = 0
        if lambda_d:
            optimizer_d.zero_grad()
            discriminator.train()

            # Train with "real" data, should get one as output
            prediction_real = discriminator(y) if not cgan else discriminator(y, x)
            loss_d_real = loss_gan(prediction_real, ones_target(N))

            # Train with "fake" data, should get zeros as output
            prediction_fake = discriminator(yhat.detach()) if not cgan else discriminator(yhat.detach(), x)
            loss_d_fake = loss_gan(prediction_fake, zeros_target(N))

            # Compute average loss
            loss_d = ((loss_d_real + loss_d_fake) / 2)
            
            # Propagate
            loss_d.backward()
            loss_d = loss_d.item()
            optimizer_d.step()





        # Return all the losses
        return loss_g.item(), loss_g_normal.item(), loss_d, loss_g_ae

    return train_step



def discriminator_shaping(generator, discriminator, loader, D, K, optimizer_d, device):

    
    loss = nn.BCELoss()

    i=0
    # for x_, y_ in loader:
    #     x_ = torch.chunk(x_, x_.size(0), dim=0) 
    #     y_ = torch.chunk(y_, y_.size(0), dim=0)
    #     for x, y in zip(x_, y_):
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if i >= D: break # Only do it for D iterations

        N = y.size(0)

        # Train with real data being our gold sample, and fake data being data that was improved by collaborative sampling
        pred_real = discriminator(y)
        pred_fake = discriminator(collaborative_sampling(generator, discriminator, x, loss, N, K))

        # Compute the loss
        loss_real = loss(pred_real, ones_target(N))
        loss_fake = loss(pred_fake, zeros_target(N))
        loss_avg = ((loss_real + loss_fake) / 2)

        # Propagate
        loss_avg.backward()
        optimizer_d.step()
