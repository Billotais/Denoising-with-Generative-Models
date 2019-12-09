from utils import ones_target, zeros_target, plot
import torch.nn as nn
import torch.optim
from progress.bar import Bar
import progressbar
import numpy as np
from test import make_test_step_gan


def train(gen, discr, loader, val, epochs, count, name, loss, optim_g, optim_d, device, gan, scheduler):


    print("Training for " + str(epochs) +  " epochs, " + str(count) + " mini-batches per epoch")
    
    train_step = make_train_step_gan(gen, discr, loss, gan, optim_g, optim_d, gan)
    test_step = make_test_step_gan(gen, discr, loss, gan)

    cuda = torch.cuda.is_available()
    losses = []
    val_losses = []
    losses_gan = []
    val_losses_gan = []
    losses_normal = []
    loss_buffer = []
    loss_normal_buffer = []
    loss_buffer_gan = []

    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_g, factor=0.5, threshold=1e-2, cooldown=0, verbose=True)
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_d, factor=0.5, threshold=1e-2, cooldown=0, verbose=True)
   
    gan_is_low = False

    for epoch in range(1, epochs+1):

        # correct the count variable if needed
        total = len(loader)
        if (total < count or count < 0): 
            count = total 
        
        #bar = progressbar.ProgressBar(max_value=count)
        curr_count = 0

        for x_batch, y_batch in loader:
            #bar.update(curr_count)
            if cuda: 
                gen.cuda()
                discr.cuda()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Train using the current batch
            loss, loss_normal, loss_gan = train_step(x_batch, y_batch, gan_is_low)

            loss_buffer.append(loss)
            loss_normal_buffer.append(loss_normal)
            loss_buffer_gan.append(loss_gan)
            # Stop if count reached
            curr_count += 1
            if (curr_count >= count): break

            # If 100 batches done
            if (len(loss_buffer) % 100 == 0):
                              
                gan_is_low = True
                # Get average train loss
                losses.append(sum(loss_buffer)/len(loss_buffer))
                losses_gan.append(sum(loss_buffer_gan)/len(loss_buffer_gan))
                losses_normal.append(sum(loss_normal_buffer)/len(loss_normal_buffer))

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, epochs, curr_count, total, losses[-1], losses_gan[-1]))
                loss_buffer = []
                loss_normal_buffer = []
                loss_buffer_gan = []

                # Compute average test loss
                val_loss_buffer = []
                val_loss_buffer_gan = []
                for x_val, y_val in val:
                    if cuda: 
                        gen.cuda()
                        discr.cuda()
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)

                    loss, loss_gan, _ = test_step(x_val, y_val)
                    val_loss_buffer.append(loss)
                    val_loss_buffer_gan.append(loss_gan)
                val_losses.append(sum(val_loss_buffer)/len(val_loss_buffer))
                val_losses_gan.append(sum(val_loss_buffer_gan)/len(val_loss_buffer_gan))
                # Every 500, plot and decrease lr in needed
                if (len(losses) % 5 == 0):
                    plot(losses, val_losses, losses_gan, val_losses_gan, losses_normal, name, gan)
                    if scheduler:
                        scheduler_g.step(val_losses[-1])
                        #scheduler_d.step(val_losses_gan[-1])
      

        # Save the model for the epoch
        torch.save({
            'gen_state_dict': gen.state_dict(),
            'optim_g_state_dict': optim_g.state_dict(),
            'discr_state_dict': discr.state_dict(),
            'optim_d_state_dict': optim_d.state_dict(),
            }, "out/" + name + "/models/model_gen_" + str(epoch) + ".tar")
        # torch.save(gen, "out/" + name + "/models/model_gen_" + str(epoch) + ".pt")
        # torch.save(discr, "out/" + name + "/models/model_discr_" + str(epoch) + ".pt")
        np.save('out/' + name + '/loss_train.npy', np.array(losses))
        np.save('out/' + name + '/loss_test.npy', np.array(val_losses))
        np.save('out/' + name + '/loss_train_gan.npy', np.array(losses_gan))
        np.save('out/' + name + '/loss_test_gan.npy', np.array(val_losses_gan))

        
    print("Model trained")
    plot(losses, val_losses, losses_gan, val_losses_gan, losses_normal, name, gan)

def make_train_step_gan(generator, discriminator, loss, lambda_d, optimizer_g, optimizer_d, GAN):

    def train_step(x, y, gan_started):


        N = y.size(0)
        loss_gan = nn.BCELoss()

        #################
        # Train generator
        #################
        optimizer_g.zero_grad()
        generator.train()
        # Generate output  distriminator
        yhat = generator(x)

        # Compute the normal loss (ususally L2)
        loss_g_normal = loss(y, yhat) 
        # Compute the adversarial loss (want want to generate realistic data)
        loss_g_adv = 0
        if GAN and gan_started:
            prediction = discriminator(yhat)
            loss_g_adv = loss_gan(prediction, ones_target(N)) 
        # Compute the global loss
        loss_g = (loss_g_normal + loss_g_normal.item()*lambda_d*loss_g_adv)
        # Propagate
        loss_g.backward()
        optimizer_g.step()

        #####################
        # Train discriminator
        #####################
        loss_d = 0
        if GAN:
            optimizer_d.zero_grad()
            discriminator.train()

            # Train with real data
            prediction_real = discriminator(y)
            loss_d_real = loss_gan(prediction_real, ones_target(N))

            # Train with fake data
            prediction_fake = discriminator(yhat.detach())
            loss_d_fake = loss_gan(prediction_fake, zeros_target(N))

            loss_d = ((loss_d_real + loss_d_fake) / 2)

            loss_d.backward()
            loss_d = loss_d.item()
            # Propagate
            optimizer_d.step()

        return loss_g.item(), loss_g_normal.item(), loss_d
    return train_step




""" def make_train_step(model, loss_fn, optimizer):

    # Builds function that performs a step in the train loop
    def train_step(x, y):
        print(x.size())
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(y, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return train_step """
