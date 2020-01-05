from utils import zeros_target, ones_target, collaborative_sampling
import torch.nn as nn
#from progress.bar import Bar
import matplotlib.pyplot as plt
import torch

def test(gen, discr, ae, loader, count, name, loss,  device, gan_lb, ae_lb, collab, cgan):

    loss = nn.MSELoss() if loss == "L2" else nn.L1Loss()
    
    test_step = make_test_step(gen, discr, ae, loss, gan_lb, ae_lb, cgan, collab)

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

def make_test_step(generator, discriminator, ae, loss_fn, gan_lb, ae_lb, cgan, collab):

    def test_step(x, y):
        N = y.size(0)
        loss = nn.BCELoss()

        # Loss of G
        generator.eval()
        yhat = generator(x)
        loss_g = loss_fn(yhat, y).item()

        # Loss of D
        loss_d = 0
        if gan_lb:
            pred = discriminator(yhat) if not cgan else discriminator(yhat, x)
            loss_d = loss(pred, zeros_target(N)).item()

        # Loss of AE
        loss_ae = 0
        if ae_lb:
            _, pred = ae(yhat)
            loss_ae = loss_fn(pred, yhat).item()

        # if collaborative gan is enabled
        if collab and gan_lb:
            # apply algo 1 for the collaborative sample
            # this way we can return an improved sample
            # maximum 50 iterations
            yhat = collaborative_sampling(generator,discriminator,x,loss,N, 50)
            
            # Apply algo 2 for Discriminator shaping 


        return loss_g, loss_d, loss_ae, yhat

    return test_step

""" def make_test_step(model, loss_fn):

def test_step(x, y):
    
    model.eval()
    yhat = model(x)

    loss = loss_fn(y, yhat)

    return loss.item(), yhat

return test_step """
