import torch
import torch.nn as nn
from peptide_uncertainty_utils import *
import random
import torch.utils.data as Data
from multi_task_heteroscedastic_regression_loss import regression_loss

class peptide_CNN(nn.Module):
    def __init__(self, num_amino_acid, max_pool_size, hidden_dim, output_dim, conv_layers, dropout_rate, kernel_nums, kernel_size):
        super(peptide_CNN,self).__init__()
        self.conv_layers = conv_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_amino_acid = num_amino_acid
        self.kernel_nums = kernel_nums
        self.kernel_size = kernel_size
        self.max_pool_size = max_pool_size
        self.convs = []
        for i in range(self.conv_layers):
            if i == 0:
                self.base_conv = nn.Sequential(nn.Conv1d(in_channels=self.num_amino_acid, 
                                                        out_channels=self.kernel_nums[i], 
                                                        kernel_size = self.kernel_size[i]),
                                                        nn.BatchNorm1d(num_features=self.kernel_nums[i]), 
                                                        nn.ReLU(),
                                                        nn.MaxPool1d(kernel_size=self.max_pool_size))
            else:
                self.base_conv = nn.Sequential(nn.Conv1d(in_channels=self.kernel_nums[i-1], 
                                                        out_channels=self.kernel_nums[i], 
                                                        kernel_size = self.kernel_size[i]),
                                                        nn.BatchNorm1d(num_features=self.kernel_nums[i]), 
                                                        nn.ReLU(),
                                                        nn.MaxPool1d(kernel_size=self.max_pool_size))
            self.convs.append(self.base_conv)
        self.convs = nn.ModuleList(self.convs)
        self.fc1 = nn.Linear(3100,self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim,self.output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,x):
        for i in range(self.conv_layers):
            x = self.convs[i](x)
        
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



class scPROTEIN_stage1_learning(nn.Module):
    def __init__(self, model, peptide_onehot_padding, Y_label,learning_rate, weight_decay, split_percentage, num_epochs,batch_size):
        super(scPROTEIN_stage1_learning, self).__init__()
        self.model = model
        self.peptide_onehot_padding = peptide_onehot_padding
        self.Y_label = Y_label
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.split_percentage = split_percentage
        self.num_epochs = num_epochs
        self.batch_size = batch_size


    def transform_peptide_data_into_dataloader(self):

        indices = list(range(self.peptide_onehot_padding.shape[0]))
        random.shuffle(indices)
        train_size = int((self.split_percentage) * len(indices))
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]
        x_train, x_valid = self.peptide_onehot_padding[train_indices], self.peptide_onehot_padding[valid_indices]
        y_train, y_valid = self.Y_label[train_indices], self.Y_label[valid_indices]
        train_dataset_split = Data.TensorDataset(x_train, y_train)
        loader = Data.DataLoader(dataset=train_dataset_split,batch_size=self.batch_size,shuffle=True)
        return loader

    def train(self):
        num_cells = self.Y_label.shape[1]
        loader = self.transform_peptide_data_into_dataloader()
        criterion = nn.MSELoss() 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, eps=1e-3)
        for epoch in range(self.num_epochs):
            self.model.train()
            loss_all = 100.
            for step, (batch_x, batch_y) in enumerate(loader):
                loss = 0.
                optimizer.zero_grad()
                y_predict = self.model(batch_x)

                for i in range(0,2*num_cells,2):
                    loss_cell = regression_loss(batch_y[:,int(i/2)], y_predict[:,i:i+2])
                    loss += loss_cell
                loss = loss/num_cells

                loss.backward()
                optimizer.step()
                loss_all += loss.item()

            self.model.eval()
            print('epoch {}, loss_regression: {}'.format(epoch, loss_all))

        # torch.save(self.model.state_dict(), 'scPROTEIN_stage1.pkl')   


    def uncertainty_generation(self):
        self.model.eval()
        y_predict_all = self.model(self.peptide_onehot_padding)
        y_predict_all = y_predict_all.cpu().detach().numpy()

        log_uncertainty = y_predict_all[:,1::2]
        uncertainty = np.exp(log_uncertainty)
        return uncertainty
        # np.save('peptide_uncertainty_tutorial.npy',uncertainty)
