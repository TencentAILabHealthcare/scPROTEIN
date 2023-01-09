import torch
import torch.nn as nn



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
