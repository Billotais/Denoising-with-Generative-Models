from utils import zeros_target, ones_target
import torch.nn as nn
from progress.bar import Bar
import matplotlib.pyplot as plt
import torch

def test(gen, discr, ae, loader, count, name, loss,  device, gan_lb, ae_lb, collab):
    test_step = make_test_step(gen, discr, ae, loss, gan_lb, ae_lb, collab)

    # Test model

    cuda = torch.cuda.is_available()
    losses = []
    outputs = []
    #bar = Bar('Testing', max=count)
    with torch.no_grad():
        for x_test, y_test in loader:
            if cuda: 
                gen.cuda()
                discr.cuda()
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            loss, _, _, y_test_hat = test_step(x_test, y_test)
            losses.append(loss)
    
            outputs.append(y_test_hat.to('cpu'))
            #bar.next()
            if (count > 0 and len(losses) >= count ): break
        #bar.finish()
    plt.plot(losses)
    plt.yscale('log')
    plt.savefig('out/'+name+'/test.png')
    plt.clf()
    return outputs

def make_test_step(generator, discriminator, ae, loss_fn, gan_lb, ae_lb, collab):

    def test_step(x, y):
        N = y.size(0)
        loss = nn.BCELoss()

        # Loss of G
        generator.eval()
        yhat = generator(x)
        loss_g = loss_fn(y, yhat).item()

        # Loss of D
        loss_d = 0
        if gan_lb:
            pred = discriminator(yhat)
            loss_d = loss(pred, zeros_target(N)).item()

        # Loss of AE
        loss_ae = 0
        if ae_lb:
            _, pred = ae(yhat)
            loss_ae = loss_fn(pred, yhat).item()

        # if the cgan is enabled, apply algo 1 for the collaborative sample
        # this way we can return an improved sample

        if collab and gan_lb:

            misclassified = True

            # Get our x_l and yhat, use last layer of generator
            xl = generator(x, lastskip=True, collab_layer=generator.depth-1, xl=None)
            yhat = generator(x)
            while misclassified:
                pred = discriminator(yhat)
                mean = pred.mean()
                if mean < 0.5: # our sample is missclassified, we have to improve it
                    # Get the gradiant of the discr loss wrt xl
                    pred = discriminator(xl)
                    loss_d = loss(pred, ones_target(N))
                    loss_d.backward()
                    # Update xl to an improved value
                    xl = xl - 0.1*xl.grad
                    # Get our new output from the generator
                    yhat = generator(x, lastskip=True, collab_layer=True, xl=xl)
                else: misclassified = False
                        

        return loss_g, loss_d, loss_ae, yhat

    return test_step

""" def make_test_step(model, loss_fn):

def test_step(x, y):
    
    model.eval()
    yhat = model(x)

    loss = loss_fn(y, yhat)

    return loss.item(), yhat

return test_step """
