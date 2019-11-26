from utils import zeros_target
import torch.nn as nn

def make_test_step_gan(generator, discriminator, loss_fn, GAN):

    def test_step(x, y):
        N = y.size(0)
        loss = nn.BCELoss()

        # Loss of G
        generator.eval()
        yhat = generator(x)
        loss_g = loss_fn(y, yhat).item()

        # Loss of D
        loss_d = 0
        if GAN:
            pred = discriminator(yhat)
            loss_d = loss(pred, zeros_target(N)).item()

        return loss_g, loss_d, yhat

    return test_step

""" def make_test_step(model, loss_fn):

def test_step(x, y):
    
    model.eval()
    yhat = model(x)

    loss = loss_fn(y, yhat)

    return loss.item(), yhat

return test_step """