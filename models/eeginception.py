import torch
import torch.nn as nn 


########################### EEG-Inception ###########################
# taken from https://github.com/esantamariavazquez/EEG-Inception/blob/main/EEGInception/EEGInception.py
# converted to PyTorch

class EEGInception(nn.Module):
    def __init__(self, 
                 input_samples=100, 
                 fs=128, 
                 n_chans=8,
                 filter_per_branch=8,
                 scales_time=(500, 250, 125), 
                 dropout_rate=0.25,
                 activation='elu'):
        super(EEGInception, self).__init__()
        
        scales_samples = [int(s * fs / 1000) for s in scales_time]
        
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError("Unsupported activation function")
        
        # Block 1
        self.b1_units = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, filter_per_branch, kernel_size=(1, scale), padding='same'),
                nn.BatchNorm2d(filter_per_branch),
                self.activation,
                nn.Dropout(dropout_rate),
                nn.Conv2d(filter_per_branch, filter_per_branch*2, 
                          kernel_size=(n_chans, 1), groups=filter_per_branch, 
                          bias=False, padding='valid'),
                nn.BatchNorm2d(filter_per_branch*2),
                self.activation,
                nn.Dropout(dropout_rate)
            )
            for scale in scales_samples
        ])
        
        # self.avg_pool1 = nn.AvgPool2d((1, 2)) 
        
        # Block 2
        self.b2_units = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(filter_per_branch * len(scales_samples) * 2, 
                          filter_per_branch, 
                          kernel_size=(1, int(scale/2)), padding='same', bias=False),
                nn.BatchNorm2d(filter_per_branch),
                self.activation,
                nn.Dropout(dropout_rate)
            )
            for scale in scales_samples
        ])
        
        # self.avg_pool2 = nn.AvgPool2d((1, 2))
        
        # Block 3
        self.b3_u1 = nn.Sequential(
            nn.Conv2d(filter_per_branch * len(scales_samples), 
                      int(filter_per_branch * len(scales_samples)/2) , 
                      kernel_size=(1, 32), padding='same', bias=False),
            
            nn.BatchNorm2d(int(filter_per_branch * len(scales_samples)/2)),
            self.activation,
            # nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.b3_u2 = nn.Sequential(
            nn.Conv2d( int(filter_per_branch * len(scales_samples)/2), 
                       int(filter_per_branch * len(scales_samples)/4),
                       kernel_size=(1, 16), padding='same', bias=False),
            
            nn.BatchNorm2d(int(filter_per_branch * len(scales_samples)/4)),
            self.activation,
            # nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout_rate)
        )
        
        # Output
        self.flatten = nn.Flatten()
        # self.dense = nn.Sequential(nn.Linear(600, 256),
        #                          nn.ReLU(),
        #                          nn.Linear(256, 1))
        self.dense = nn.Linear(600, 1)
        
    def forward(self, x):
        # Block 1
        b1_out = torch.cat([unit(x) for unit in self.b1_units], dim=1)
        # Block 2
        b2_out = torch.cat([unit(b1_out) for unit in self.b2_units], dim=1)
        # Block 3
        b3_u1_out = self.b3_u1(b2_out)
        b3_u2_out = self.b3_u2(b3_u1_out)
        # Flatten and output
        embeddings = self.flatten(b3_u2_out)
        return embeddings, self.dense(embeddings)
    
    
########################### EEG-Inception Original ###########################


class EEGInception_Original(nn.Module):
    def __init__(
        self,
        fs=128,
        n_chans=8,
        filters_per_branch=8,
        scales_time=(500, 250, 125),
        dropout_rate=0.25,
        activation=nn.ELU(inplace=True),
    ):
        super().__init__()
        scales_samples = [int(s * fs / 1000) for s in scales_time]
        # ========================== BLOCK 1: INCEPTION ========================== #
        self.inception1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        1,
                        filters_per_branch,
                        (1, scales_sample),
                        padding="same",
                    ),
                    nn.BatchNorm2d(filters_per_branch),
                    activation,
                    nn.Dropout(dropout_rate),
                    nn.Conv2d(filters_per_branch, filters_per_branch*2, 
                          kernel_size=(n_chans, 1), groups=filters_per_branch, 
                          bias=False, padding='valid'),
                    nn.BatchNorm2d(filters_per_branch * 2),
                    activation,
                    nn.Dropout(dropout_rate),
                )
                for scales_sample in scales_samples
            ]
        )
        self.avg_pool1 = nn.AvgPool2d((1, 4))

        # ========================== BLOCK 2: INCEPTION ========================== #
        self.inception2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        len(scales_samples) * 2 * filters_per_branch,
                        filters_per_branch,
                        (1, scales_sample // 4),
                        bias=False,
                        padding="same",
                    ),
                    nn.BatchNorm2d(filters_per_branch),
                    activation,
                    nn.Dropout(dropout_rate),
                )
                for scales_sample in scales_samples
            ]
        )

        self.avg_pool2 = nn.AvgPool2d((1, 2))

        # ============================ BLOCK 3: OUTPUT =========================== #
        self.output = nn.Sequential(
            nn.Conv2d(
                filters_per_branch * len(scales_samples),
                filters_per_branch * len(scales_samples) // 2,
                (1, 8),
                bias=False,
                padding="same",
            ),
            nn.BatchNorm2d(filters_per_branch * len(scales_samples) // 2),
            activation,
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                filters_per_branch * len(scales_samples) // 2,
                filters_per_branch * len(scales_samples) // 4,
                (1, 4),
                bias=False,
                padding="same",
            ),
            nn.BatchNorm2d(filters_per_branch * len(scales_samples) // 4),
            activation,
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout_rate),
        )
        self.dense = nn.Linear(18, 1)

    def forward(self, x):
        x = torch.cat([net(x) for net in self.inception1], dim=1)
        x = self.avg_pool1(x)
        x = torch.cat([net(x) for net in self.inception2], dim=1)
        x = self.avg_pool2(x)
        x = self.output(x)
        x = torch.flatten(x, start_dim=1)
        return x,  self.dense(x)