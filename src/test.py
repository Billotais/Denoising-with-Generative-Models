from utils import zeros_target
import torch.nn as nn
from progress.bar import Bar
import matplotlib.pyplot as plt
import torch

def test(gen, discr, loader, count, name, loss,  device, gan):
    test_step = make_test_step_gan(gen, discr, loss, gan)

    # Test model

    cuda = torch.cuda.is_available()
    losses = []
    outputs = []
    bar = Bar('Testing', max=count)
    with torch.no_grad():
        for x_test, y_test in loader:
            if cuda: 
                gen.cuda()
                discr.cuda()
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            loss, _, y_test_hat = test_step(x_test, y_test)
            losses.append(loss)
    
            outputs.append(y_test_hat.to('cpu'))
            bar.next()
            if (count > 0 and len(losses) >= count ): break
        bar.finish()
    plt.plot(losses)
    plt.yscale('log')
    plt.savefig('out/'+name+'/test.png')
    plt.clf()
    return outputs

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