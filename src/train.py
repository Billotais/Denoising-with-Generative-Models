from utils import ones_target, zeros_target
import torch.nn as nn

def make_train_step_gan(generator, discriminator, loss, lambda_d, optimizer_g, optimizer_d, GAN):

    def train_step(x, y):

        #scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.5)

        N = y.size(0)
        loss_gan = nn.BCELoss()

        #################
        # Train generator
        #################
        optimizer_g.zero_grad()
        generator.train()

        # Generate output  distriminator
        yhat = generator(x)


        # Compute the normal loss
        loss_g_normal = loss(y, yhat) 
        # Compute the adversarial loss (want want to generate realistic data)
        loss_g_adv = 0
        if GAN:
            prediction = discriminator(yhat)
            loss_g_adv = loss_gan(prediction, ones_target(N)) 
        # Compute the global loss
        loss_g = (loss_g_normal + loss_g_normal.item()*lambda_d*loss_g_adv)

        # Propagate
        loss_g.backward(retain_graph=True)
        optimizer_g.step()
        #scheduler_g.step()

        loss_g = loss_g.item()

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
            loss_d_real.backward(retain_graph=True)

            # Train with fake data
            
            prediction_fake = discriminator(yhat)
            loss_d_fake = loss_gan(prediction_fake, zeros_target(N))
            loss_d_fake.backward()

            loss_d = ((loss_d_real + loss_d_fake) / 2).item()

            # Propagate
            optimizer_d.step()

        return loss_g, loss_d
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