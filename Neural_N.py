"""
Reconstruction Algorithim to predict the primordial power spectrum from Planck data

Machine learning model using pytorch that generates a training dataset, trains a neural network and
predicts Pk from observational data from Planck mission


This module allows you to train and make predictions with three different expresions of Pk.


Possible mode options:

-Power_law               Original expression of Pk
-Fourier                 Fourier Series perturbation of the original formula
-Polynomial              Fifth degree olynomial perturbation of the original formula
-All                     Create a dataset made up with a colection of the three expressions together



Pablo Cuadrado
University of sussex
pcuadradolo97@gmail.com
"""


"""
Imports of needed libraries
"""
import camb
import numpy as np
from camb import model, initialpower
from random import seed
import random
import torch
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from torch.nn import  MSELoss



from torch.utils.data import DataLoader,Dataset
from tqdm.auto import trange
import os

from matplotlib import ticker


def get_results():
    """
    Calculate the fiducial transfer function

    """

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)  # 0.9- 1,1  1-3 e-9
    pars.set_for_lmax(2500, lens_potential_accuracy=0);

    results = camb.get_results(pars)

    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    return results


def update_results(k, pk, results):

    """
    Update a CAMB data object to a new primordial power spectrum

    """

    inflation_params = initialpower.SplinedInitialPower()
    inflation_params.set_scalar_table(k, pk)
    results.power_spectra_from_transfer(inflation_params)
    return results


def get_cmb_cls(results):

    """"
    Get the CMB TT Correlation function from a CAMBdata object

    """

    cl = results.get_total_cls(2500, CMB_unit='muK')
    return cl[2:, 0]



def Fourier(k, a, b):

    """
    Calculate a Fourier Series perturbation
    """

    for n in range(len(a)):
        f = 1 + a[n] * np.cos(2 * np.pi * n * k*10) + b[n] * np.sin(2 * np.pi * n * k*10)

    return f


def Poly(k):

    """
    Calculate a fifth degree polynomial perturbation
    """
    a=np.random.rand(1)
    b=np.random.rand(1)
    c=np.random.rand(1)
    d=np.random.rand(1)
    e=np.random.rand(1)

    f=1+a*k + b*(k**2) + c*(k**3) + d*(k**4) + e*(k**5)

    return f



def Power_Spec(k, As, ns, a, b):

    """
    Calculate primordial power spectrum returning:

    Power-law
    Fourier Series perturbation
    Fifth degree polynomial perturbation
    """

    pk_0 = As * (k ** (ns - 1))

    return pk_0, pk_0 * Fourier(k, a, b),pk_0*Poly(k)


def Cl_plot(k,mode,pk0,pk):

    """
    Plot Cl for teo different Pk functions
    """

    results = get_results()
    results = update_results(k, pk0, results)

    cls1 = get_cmb_cls(results)

    results = update_results(k, pk, results)
    cls2 = get_cmb_cls(results)


    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    ax0 = plt.subplot(gs[0])
    plt.title('CMB Power Spectrum',fontsize=15)
    line1, = ax0.plot(cls1, color='r', label='Power-law')
    line2, = ax0.plot(cls2, label=mode+' perturbation')
    plt.ylabel('$D_{\ell}$ $[\mu K^{2}]$',fontsize=12)
    plt.xlabel('$\ell$',fontsize=12)
    ax0.legend(fontsize=15)

    ax1 = plt.subplot(gs[1], sharex=ax0)
    line3, = plt.plot(cls1 - cls2, '--', label='Cls1-cls2')
    plt.subplots_adjust(hspace=.0)
    plt.ylabel('$\Delta D_{\ell}$',fontsize=12)
    plt.xlabel('$\ell$ ',fontsize=15)
    plt.show()



def Power_Spec_Plot(k,mode,pk0,pk):

    """
    Plot primordial power spectrum functions
    """


    fig, ax = plt.subplots()

    plt.plot(k, pk0*(10**9) , color='r', label='Power Law')
    plt.plot(k, pk*(10**9) , label=mode+ ' perturbation')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$k$ $(Mpc^{-1})$', labelpad=8, fontsize=12)
    plt.ylabel('$P_{k}$ ($\cdot10^{9})$', labelpad=8, fontsize=12)

    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())


    plt.title('Primordial power spectrum', fontsize=15)
    plt.legend(fontsize=15)

    plt.show()



def Plots(mode):

    """
    Plot of both Cl and Pk
    """


    a = np.random.uniform(0, 1, 10) / 10
    b = np.random.uniform(0, 1, 10) / 10

    As = random.uniform(1e-9, 3e-9)
    ns = random.uniform(0.9, 1.1)

    logk = np.linspace(-5, -1.1, 1000)
    k = 10 ** logk

    pk=Power_Spec(k,As,ns,a,b)


    if mode=='Fourier':

        Power_Spec_Plot(k,mode,pk[0],pk[1])
        Cl_plot(k,mode,pk[0],pk[1])

    elif mode=='Polynomial':

        Power_Spec_Plot(k,mode, pk[0], pk[2])
        Cl_plot(k,mode,pk[0], pk[2])



    else:
        print(
            'Mode no valid!\n \nPlease introduce one of the following options:   \n Fourier \n Polynomial ')
        quit()




def Cl_final(k,mode):

    """
    Calculate Pk and Cl for each possible mode (or expression of Pk). Adds noise to Cl function
    """


    a = np.random.uniform(0, 1, 10) / 10
    b = np.random.uniform(0, 1, 10) / 10

    As = random.uniform(1e-9, 3e-9)
    ns = random.uniform(0.9, 1.1)

    pk = Power_Spec(k, As, ns, a, b)

    simulator = CMBNativeSimulator(use_cl=['tt'])

    if mode=='Power_law':

        results = get_results()
        results = update_results(k, pk[0], results)

        cls1 = get_cmb_cls(results)

        noiseless_cl = simulator.get_cl(30, cls1)
        noisy_cl = simulator.get_noisy_cl(30, cls1)

        Dl=noisy_cl* simulator.lav * (simulator.lav - 1) / 2 / np.pi

        return Dl,pk[0]

    elif mode=='Fourier':

        results = get_results()
        results = update_results(k, pk[1], results)

        cls1 = get_cmb_cls(results)


        noiseless_cl = simulator.get_cl(30, cls1)
        noisy_cl = simulator.get_noisy_cl(30, cls1)

        Dl=noisy_cl* simulator.lav * (simulator.lav - 1) / 2 / np.pi

        return Dl, pk[1]


    elif mode=='Polynomial':

        results = get_results()
        results = update_results(k, pk[2], results)

        cls1 = get_cmb_cls(results)

        noiseless_cl = simulator.get_cl(30, cls1)
        noisy_cl = simulator.get_noisy_cl(30, cls1)

        Dl = noisy_cl * simulator.lav * (simulator.lav - 1) / 2 / np.pi

        return Dl, pk[2]


    elif mode == 'All':



        random_n = random.randint(1, 3)


        results = get_results()
        results = update_results(k, pk[random_n - 1], results)

        cls1 = get_cmb_cls(results)

        noiseless_cl = simulator.get_cl(30, cls1)
        noisy_cl = simulator.get_noisy_cl(30, cls1)

        cl = noisy_cl * simulator.lav * (simulator.lav - 1) / 2 / np.pi

        return cl, pk[random_n - 1]


    else:
        print(
            'Mode no valid!\n \nPlease introduce one of the following options: \n Power_law  \n Fourier \n Polynomial \n All')
        quit()





def DataGenerator(N,mode):

    """
    Prepare data to create training set. Normalize both input and output
    """

    cls = []
    pks = []
    pk_max = []
    pk_min = []
    Cl_max=[]
    Cl_min=[]

    logk = np.linspace(-5, -1.1, 215)
    k = 10 ** logk

    for i in range(N):
        y = Cl_final(k, mode)

        cls.append(y[0])
        pks.append(y[1])


    pks = np.array(pks)
    cls=np.array(cls)


    for i in range(len(k)):

        pk_max.append(pks.transpose()[i].max())
        pk_min.append(pks.transpose()[i].min())

    pk_max = np.array(pk_max)
    pk_min = np.array(pk_min)

    for i in range(int(cls.size/N)):

        Cl_max.append(cls.transpose()[i].max())
        Cl_min.append(cls.transpose()[i].min())

    Cl_max=np.array(Cl_max)
    Cl_min=np.array(Cl_min)



    for i in range(N):

        pks[i]=(pks[i]-pk_min)/(pk_max-pk_min)
        cls[i]=(cls[i]-Cl_min)/(Cl_max-Cl_min)


    return cls, pks,pk_max,pk_min,Cl_max,Cl_min




def dataset(Data_size,mode):

    """
    Load and prepare data set for training
    """

    device='cuda:0' if torch.cuda.is_available() else 'cpu'

    Cl,pk,pk_max,pk_min,Cl_max,Cl_min=DataGenerator(Data_size,mode)

    Cl=torch.tensor(Cl, dtype=torch.float32, device=device)
    pk=torch.tensor(pk, dtype=torch.float32, device=device)

    Cl=Cl.to(device=device)
    pk=pk.to(device=device)

    dataset = torch.utils.data.TensorDataset(Cl, pk)



    """
    Splitting data into train and test set:
    
    80% of dataset: Train set
    20% of dataset: Test set
    """

    n_test = int(0.2 * len(Cl))
    n_train = len(Cl) - n_test

    train_set, validation_set = torch.utils.data.random_split(dataset, [n_train, n_test])

    return train_set,validation_set,pk_max,pk_min,Cl_max,Cl_min


def load_planck():

    """"
    Function to load and read the Planck data
    """

    with open('PlanckData.txt') as f:
        lines = f.readlines()
        l = []
        Dl = []
        Error_inf = []
        Error_sup = []
        for line in lines:
            line = line.split()
            line = [float(i) for i in line]
            l.append(line[0])
            Dl.append(line[1])
            Error_inf.append(line[2])
            Error_sup.append(line[3])

    return np.array(Dl)

def Planclk_Plot():

    """
    Load and plot Planck data
    """

    with open('PlanckBinned.txt') as f:
        lines = f.readlines()
        l_b = []
        Dl_b = []
        Error_inf_b = []
        Error_sup_b = []
        fit = []
        for line in lines:
            line = line.split()
            line = [float(i) for i in line]
            l_b.append(line[0])
            Dl_b.append(line[1])
            Error_inf_b.append(line[2])
            Error_sup_b.append(line[3])
            fit.append(line[4])

    plt.plot(l_b,Dl_b,color='b')
    plt.errorbar(l_b,Dl_b,(Error_inf_b,Error_sup_b),fmt='r.')
    plt.title('Planck CMB spectrum',fontsize=15)
    plt.xlabel('Multipole moment $\ell$',fontsize=15)
    plt.ylabel('$D_{\ell}$ $[\mu k^{2}]$',fontsize=15)
    plt.plot(l_b,fit)

    plt.show()





def Training(model,mode,n_epochs):

    """
    Training function:

    Hyperparameters that define the learning rate of the neural network.
    """

    device='cuda:0' if torch.cuda.is_available() else 'cpu'

    loss_fn = MSELoss()

    learning_rate = 0.01

    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

    batch_size = 5

    Data_size = 100


    n_epochs = n_epochs



    """ Load and split the data into a train set and test set. The pk and Cl maximum and minimum values are needed to reconstruct
    the normalization of the dataset """



    train_set, val_set,pk_max,pk_min,Cl_max,Cl_min = dataset(Data_size,mode)


    trainloader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size,
                                              shuffle=True)

    val_loader=torch.utils.data.DataLoader(val_set,
                                           batch_size=batch_size,
                                           shuffle=True)



    """ Training loop over the number of epochs.It calculates the loss of the train and test set per each epoch """


    t = trange(n_epochs, desc='Batch', leave=True)
    for epoch in t:
        train_loss = []
        test_loss = []
        for Cl, pk in trainloader:
            pk_pred = model(Cl)
            loss = loss_fn(pk_pred, pk)

            model.zero_grad()
            loss.backward()
            opt.step()

            train_loss.append(loss.item())


        t.set_description(f"Epoch {epoch}")

        for Cl,pk in val_loader:
            model.eval()

            pk_pred=model(Cl)
            loss=loss_fn(pk_pred,pk)
            test_loss.append(loss.item())


        t.set_postfix(train_loss=np.average(train_loss), test_loss=np.average(test_loss))



    return pk_max,pk_min,Cl_max,Cl_min




def Plot_training(mode,n_epochs):

    """
    Function to plot the results after the training and make predictions using planck data

    ------

    Neural network using torch.nn.Sequential and ReLU as activation function
    """

    model = torch.nn.Sequential(
        torch.nn.Linear(215, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 215),
    )


    """ Training of the model """

    pk_max,pk_min,Cl_max,Cl_min=Training(model,mode,n_epochs)



    """ Load and prepare unknown data for the model to plot  prediction and see how the model performs """

    logk = np.linspace(-5, -1.1, 215)
    k = 10 ** logk


    if mode=='All':

        mode='Power_law'
        y = Cl_final(k, mode)

        Cl = (y[0] - Cl_min) / (Cl_max - Cl_min)
        pk = (y[1] - pk_min) / (pk_max - pk_min)

        model.eval()

        Cl = torch.tensor(Cl, dtype=torch.float32, device='cuda:0' if torch.cuda.is_available() else 'cpu')

        pk_pred = model(Cl)

        """ Plot of both expected and predicted value for Cl """

        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

        fig1 = plt.figure()

        ax0 = plt.subplot(gs[0])
        plt.title('Primordial Power Spectrum', fontsize=15)
        line1, = ax0.plot(k, pk * (pk_max - pk_min) + pk_min, color='r', label='Expected')
        line2, = ax0.plot(k, pk_pred.detach().numpy() * (pk_max - pk_min) + pk_min, label='Predicted')
        plt.ylabel('$P_{k}$', fontsize=15)

        plt.xscale('log')
        ax0.legend(fontsize=15)

        ax1 = plt.subplot(gs[1], sharex=ax0)
        line3, = plt.plot(k, np.abs(pk - pk_pred.detach().numpy()), '--', label='Cls1-cls2')
        plt.subplots_adjust(hspace=.0)
        plt.ylabel('$\Delta P_{k}$', fontsize=12)
        plt.xlabel('$k$ $(Mpc^{-1})$', fontsize=10)
        plt.xscale('log')
        #plt.show()
        fig1.savefig(mode+'_train.png')
        plt.close(fig1)



        mode='Polynomial'
        y = Cl_final(k, mode)

        Cl = (y[0] - Cl_min) / (Cl_max - Cl_min)
        pk = (y[1] - pk_min) / (pk_max - pk_min)

        model.eval()

        Cl = torch.tensor(Cl, dtype=torch.float32, device='cuda:0' if torch.cuda.is_available() else 'cpu')

        pk_pred = model(Cl)

        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

        fig1 = plt.figure()

        ax0 = plt.subplot(gs[0])
        plt.title('Primordial Power Spectrum', fontsize=15)
        line1, = ax0.plot(k, pk * (pk_max - pk_min) + pk_min, color='r', label='Expected')
        line2, = ax0.plot(k, pk_pred.detach().numpy() * (pk_max - pk_min) + pk_min, label='Predicted')
        plt.ylabel('$P_{k}$', fontsize=15)

        plt.xscale('log')
        ax0.legend(fontsize=15)

        ax1 = plt.subplot(gs[1], sharex=ax0)
        line3, = plt.plot(k, pk - pk_pred.detach().numpy(), '--', label='Cls1-cls2')
        plt.subplots_adjust(hspace=.0)
        plt.ylabel('$\Delta P_{k}$', fontsize=12)
        plt.xlabel('$k$ $(Mpc^{-1})$', fontsize=10)
        plt.xscale('log')
        #plt.show()
        fig1.savefig(mode + '_train.png')
        plt.close(fig1)



        mode='Fourier'
        y = Cl_final(k, mode)
        Cl = (y[0] - Cl_min) / (Cl_max - Cl_min)
        pk = (y[1] - pk_min) / (pk_max - pk_min)

        model.eval()

        Cl = torch.tensor(Cl, dtype=torch.float32, device='cuda:0' if torch.cuda.is_available() else 'cpu')

        pk_pred = model(Cl)

        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

        fig1 = plt.figure()

        ax0 = plt.subplot(gs[0])
        plt.title('Primordial Power Spectrum', fontsize=15)
        line1, = ax0.plot(k, pk * (pk_max - pk_min) + pk_min, color='r', label='Expected')
        line2, = ax0.plot(k, pk_pred.detach().numpy() * (pk_max - pk_min) + pk_min, label='Predicted')
        plt.ylabel('$P_{k}$', fontsize=15)

        plt.xscale('log')
        ax0.legend(fontsize=15)

        ax1 = plt.subplot(gs[1], sharex=ax0)
        line3, = plt.plot(k, np.abs(pk - pk_pred.detach().numpy()), '--', label='Cls1-cls2')
        plt.subplots_adjust(hspace=.0)
        plt.ylabel('$\Delta P_{k}$', fontsize=12)
        plt.xlabel('$k$ $(Mpc^{-1})$', fontsize=10)
        plt.xscale('log')
        #plt.show()
        fig1.savefig(mode + '_train.png')
        plt.close(fig1)

    else:

        y = Cl_final(k, mode)

        Cl = (y[0] - Cl_min) / (Cl_max - Cl_min)
        pk = (y[1] - pk_min) / (pk_max - pk_min)

        model.eval()

        Cl = torch.tensor(Cl, dtype=torch.float32, device='cuda:0' if torch.cuda.is_available() else 'cpu')

        pk_pred = model(Cl)

        """ Plot of both expected and predicted value for Cl """

        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

        fig1 = plt.figure()

        ax0 = plt.subplot(gs[0])
        plt.title('Primordial Power Spectrum', fontsize=15)
        line1, = ax0.plot(k, pk * (pk_max - pk_min) + pk_min, color='r', label='Expected')
        line2, = ax0.plot(k, pk_pred.detach().numpy() * (pk_max - pk_min) + pk_min, label='Predicted')
        plt.ylabel('$P_{k}$', fontsize=15)

        plt.xscale('log')
        ax0.legend(fontsize=15)

        ax1 = plt.subplot(gs[1], sharex=ax0)
        line3, = plt.plot(k, np.abs(pk - pk_pred.detach().numpy()), linestyle='dotted', label='Cls1-cls2')
        plt.subplots_adjust(hspace=.0)
        plt.ylabel('$\Delta P_{k}$', fontsize=12)
        plt.xlabel('$k$ $(Mpc^{-1})$', fontsize=10)
        plt.xscale('log')
        #plt.show()
        fig1.savefig(mode + '_train.png')
        plt.close(fig1)




    """ Load and prepare Planck Data to plot prediction """

    Dl = load_planck()

    Dl = np.array(Dl[0:2499])

    simulator = CMBNativeSimulator(use_cl=['tt'])
    noiseless_cl = simulator.get_cl(30, Dl)
    noisy_cl = simulator.get_noisy_cl(30, Dl)

    cl = noisy_cl * simulator.lav * (simulator.lav - 1) / 2 / np.pi
    Dl = (cl - Cl_min) / (Cl_max - Cl_min)

    Dl = torch.tensor(Dl, dtype=torch.float32, device='cuda:0' if torch.cuda.is_available() else 'cpu')

    pk_pred = model(Dl)





    fig2,ax=plt.subplots()
    plt.plot(k, (pk_pred.detach().numpy() * (pk_max - pk_min) + pk_min)*(10**9))
    plt.xscale('log')
    plt.yscale('log')
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.title(mode+' prediction from Planck', fontsize=15)
    plt.xlabel('$k$ $(Mpc^{-1})$ ', fontsize=10)
    plt.ylabel('$P_{k}$ $(\cdot 10^{9})$', fontsize=12)
    #plt.show()
    fig2.savefig(mode+'_predss.png')
    plt.close(fig2)


    with open(str(n_epochs)+'.txt','w+') as f:
        np.savetxt(f,np.array(pk_pred.detach().numpy() * (pk_max - pk_min) + pk_min), delimiter=', ')





def Compare_preds():

    """
    Train and plot predictions for the three different expressions for Pk
    """

    pks = ['Power_law', 'Fourier', 'Polynomial']


    Plot_training(mode='Power_law')

    Plot_training(mode='Fourier')

    Plot_training(mode='Polynomial')

    with open('Power_law.txt') as f:
        lines = f.readlines()
        pk_power_law = []
        for line in lines:
            line = line.split()
            line = [float(i) for i in line]
            pk_power_law.append(line[0])


    with open('Fourier.txt') as f:
        lines = f.readlines()
        pk_fourier = []
        for line in lines:
            line = line.split()
            line = [float(i) for i in line]
            pk_fourier.append(line[0])


    with open('Polynomial.txt') as f:
        lines = f.readlines()
        pk_poly = []
        for line in lines:
            line = line.split()
            line = [float(i) for i in line]
            pk_poly.append(line[0])


    logk = np.linspace(-5, -1.1, 300)
    k = 10 ** logk
    fig3=plt.figure()
    plt.plot(k,pk_power_law,label='Power-law')
    plt.plot(k,pk_fourier,label='Fourier')
    plt.plot(k,pk_poly,label='Polynomial')
    plt.title('Primordial Power Spectrum')
    plt.xlabel('$k (Mpc^{-1})$ ', fontsize=10)
    plt.ylabel('$P_{k}$', fontsize=12)
    plt.xscale('log')
    plt.legend()
    #plt.show()
    fig3.savefig('Predictions.png')



class CMBNativeSimulator():

    """
    Convert unbinned Cl to binned Cl and ads noise to the output
    """

    def __init__(self, use_cl=['tt', 'te', 'ee'], path="./data"):
        nbintt = 215
        nbinte = 199
        nbinee = 199
        self.use_cl = use_cl
        cl_names = ['tt', 'te', 'ee']

        lmax = 2508
        use_bins = []
        bins_for_L_range = []

        data = np.loadtxt(os.path.join(path, "cl_cmb_plik_v22.dat"))

        bin_lmin_offset = 30
        self.blmin = np.loadtxt(os.path.join(path, 'blmin.dat')).astype(
            int) + bin_lmin_offset
        self.blmax = np.loadtxt(os.path.join(path, 'blmax.dat')).astype(
            int) + bin_lmin_offset
        self.lav = (self.blmin + self.blmax) // 2
        weights = np.loadtxt(os.path.join(path, 'bweight.dat'))
        ls = np.arange(len(weights)) + bin_lmin_offset
        self.bin_lmin_offset = bin_lmin_offset
        weights *= 2 * np.pi / ls / (ls + 1)  # we work directly with  DL not CL
        self.weights = np.hstack((np.zeros(bin_lmin_offset), weights))

        self.nbins = nbintt + nbinee + nbinte

        bin_cov_file = os.path.join(path, 'c_matrix_plik_v22.dat')

        if os.path.exists(bin_cov_file):
            from scipy.io import FortranFile
            f = FortranFile(bin_cov_file, 'r')
            cov = f.read_reals(dtype=float).reshape((self.nbins, self.nbins))
            cov = np.tril(cov) + np.tril(cov, -1).T  # make symmetric
        else:
            raise Exception("Covariance matrix not found")
            # cov = np.loadtxt(ini.relativeFileName('cov_file'))
            # full n row x n col matrix converted from fortran binary

        self.lmax = lmax

        maxbin = max(nbintt, nbinte, nbinee)
        assert (cov.shape[0] == self.nbins)
        self.lav = self.lav[:maxbin]

        if len(use_bins) and np.max(use_bins) >= maxbin:
            raise ValueError('use_bins has bin index out of range')
        if len(bins_for_L_range):
            if len(use_bins):
                raise ValueError('can only use one bin filter')
            use_bins = [use_bin for use_bin in range(maxbin)
                        if
                        bins_for_L_range[0] <= (
                                self.blmin[use_bin] + self.blmax[use_bin]) / 2 <=
                        bins_for_L_range[1]]
            print('Actual L range: %s - %s' % (
                self.blmin[use_bins[0]], self.blmax[use_bins[-1]]))

        self.used = np.zeros(3, dtype=bool)
        self.used_bins = []
        used_indices = []
        offset = 0
        self.bandpowers = {}
        self.errors = {}

        for i, (cl, nbin) in enumerate(zip(cl_names, [nbintt, nbinte, nbinee])):
            self.used[i] = cl_names[i] in self.use_cl
            sc = self.lav[:nbin] * (self.lav[:nbin] + 1) / 2. / np.pi
            self.bandpowers[cl] = data[offset:offset + nbin, 1] * sc
            self.errors[cl] = data[offset:offset + nbin, 2] * sc
            if self.used[i]:
                if len(use_bins):
                    self.used_bins.append(
                        np.array([use_bin for use_bin in use_bins if use_bin < nbin],
                                 dtype=int))
                else:
                    self.used_bins.append(np.arange(nbin, dtype=int))
                used_indices.append(self.used_bins[-1] + offset)
            else:
                self.used_bins.append(np.arange(0, dtype=int))
            offset += nbin
        self.used_indices = np.hstack(used_indices)
        assert (self.nbins == cov.shape[0] == data.shape[0])
        self.X_data = data[self.used_indices, 1]
        self.cov = cov[np.ix_(self.used_indices, self.used_indices)]
        self.invcov = np.linalg.inv(self.cov)

    def get_cl(self, L0, ctt, calPlanck=1):
        """
        Returns the CMB power spectrum at L > L0, given the input Cls.
        """
        cl = np.empty(self.used_indices.shape)
        #ctt = totCL[:, 0]
        #cte = totCL[:, 3]
        #cee = totCL[:, 1]
        #ctt = totCL[:, 0]
        cte = 0
        cee = 0
        ix = 0
        for tp, cell in enumerate([ctt, cte, cee]):
            for i in self.used_bins[tp]:
                cl[ix] = np.dot(cell[self.blmin[i] - L0:self.blmax[i] - L0 + 1],
                                self.weights[self.blmin[i]:self.blmax[i] + 1])
                ix += 1
        cl /= calPlanck ** 2
        return cl

    def get_noisy_cl(self, L0, totCL, calPlanck=1):
        """
        Returns a noisy CMB power spectrum at L > L0, given the input Cls.
        """
        cl = self.get_cl(L0, totCL, calPlanck=1)
        return np.random.multivariate_normal(cl, self.cov)



def Noisy():

    """
    Plot noiseless and noisy Dl function
    """

    logk = np.linspace(-5, -1.1, 215)
    k = 10 ** logk

    a = np.random.uniform(0, 1, 10) / 10
    b = np.random.uniform(0, 1, 10) / 10

    As = random.uniform(1e-9, 3e-9)
    ns = random.uniform(0.9, 1.1)

    pk = Power_Spec(k, As, ns, a, b)

    simulator = CMBNativeSimulator(use_cl=['tt'])

    results = get_results()
    results = update_results(k, pk[0], results)

    Cl = get_cmb_cls(results)


    noiseless_cl=simulator.get_cl(30,Cl)
    noisy_cl=simulator.get_noisy_cl(30,Cl)

    noiseless_Dl=noiseless_cl* simulator.lav * (simulator.lav - 1) / 2 / np.pi
    noisy_Dl=noisy_cl* simulator.lav * (simulator.lav - 1) / 2 / np.pi

    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    fig1 = plt.figure()

    ax0 = plt.subplot(gs[0])
    plt.title('$D_{\ell}$ noisy realisations', fontsize=15)
    line2, = ax0.plot(simulator.lav, noisy_Dl, color='r', label='Noisy $D_{\ell}$')
    line1, = ax0.plot(simulator.lav,noiseless_Dl,label='Noiseless $D_{\ell}$')
    plt.ylabel('$D_{\ell}$ $[\mu K^{2}]$', fontsize=15)


    ax0.legend(fontsize=15)

    ax1 = plt.subplot(gs[1], sharex=ax0)
    line3, = plt.plot(simulator.lav, noiseless_Dl - noisy_Dl, linestyle='dotted', label='Cls1-cls2')
    plt.subplots_adjust(hspace=.0)
    plt.ylabel('$\Delta D_{\ell}$', fontsize=12)
    plt.xlabel('$\ell$ ', fontsize=15)
    plt.show()






def Final_Prediction(mode,n_epochs):

    """
    Train the model and make predictions fro diferent noisy realisations for Dl
    """

    model = torch.nn.Sequential(
        torch.nn.Linear(215, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 215),
    )



    """ Training of the model """

    pk_max, pk_min, Cl_max, Cl_min = Training(model, mode, n_epochs)

    Dl = load_planck()

    Dl = np.array(Dl[0:2499])


    simulator = CMBNativeSimulator(use_cl=['tt'])



    model.eval()

    fig, ax = plt.subplots()

    logk = np.linspace(-5, -1.1, 215)
    k = 10 ** logk

    """
    Generate different noisy Dl function and make prediction for each one to plot them all together
    """

    for i in range(10):

        noisy_Cl = simulator.get_noisy_cl(30, Dl) * simulator.lav * (simulator.lav - 1) / 2 / np.pi
        noisy_Cl_norm = (noisy_Cl - Cl_min) / (Cl_max - Cl_min)
        noisy_Dl = torch.tensor(noisy_Cl_norm, dtype=torch.float32,device='cuda:0' if torch.cuda.is_available() else 'cpu')
        pk_pred = model(noisy_Dl)

        plt.plot(k, (pk_pred.detach().numpy() * (pk_max - pk_min) + pk_min)*(10**9),alpha=0.6)


    plt.xscale('log')
    plt.yscale('log')
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.title('Reconstruction from Planck',fontsize=15)
    plt.xlabel('$k$ $(Mpc^{-1})$',fontsize=12)
    plt.ylabel('$P_{k}$ $(\cdot 10^{9})$',fontsize=12)
    plt.show()



def plot_epochs(mode):

    """
    Train and predict for different number of epochs in the same
    """

    epochs = [5000,10000,15000,20000]

    fig4 = plt.figure()
    logk = np.linspace(-5, -1.1, 300)
    k = 10 ** logk


    for i in epochs:

        Plot_training(mode, n_epochs=i)

        with open(str(i) + '.txt') as f:
            lines = f.readlines()
            pk=[]
            for line in lines:
                line = line.split()
                line = [float(i) for i in line]
                pk.append(line[0])

        plt.plot(k, pk, label=str(i) + ' epochs')

    plt.xscale('log')
    plt.legend()
    fig4.savefig('epochs.png')



if __name__ == "__main__":


    """
    To generate plots for Pk and Cl
    
    Introduce one mode:
    
    mode='Fourier' 
    mode='Polynomial'
    """
    #mode=''
    #Plots(mode)



    """
    To plot Planck data
    """
    #Planclk_Plot()


    """
    To train the model and plot its performance and prediction
    
    Introduce n_epochs and one mode:
    
    mode='Power_law'
    mode='Fourier' 
    mode='Polynomial'
    mode='All'
    """

    #mode=''
    #n_epochs=
    #Plot_training(mode,n_epochs)



    """
    To train and plot predictions
    
    Introduce n_epochs and one mode:
    
    mode='Power_law'
    mode='Fourier' 
    mode='Polynomial'
    mode='All'
    """

    #mode=''
    #n_epochs=
    #Final_Prediction(mode,n_epochs)

