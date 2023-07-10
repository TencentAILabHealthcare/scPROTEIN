import torch




# multi-task heteroscedastic regression loss
def regression_loss(y_true, y_pred):

    y_true = y_true.reshape(-1,1)
    mu = y_pred[:, :1] # first output neuron: predicted expression value
    log_sig = y_pred[:, 1:] # second output neuron: uncertainty estimation
    sig = torch.exp(log_sig) 

    return torch.mean(2*log_sig + ((y_true-mu)/sig)**2) 


