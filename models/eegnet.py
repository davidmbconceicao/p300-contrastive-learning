import torch.nn as nn 

########################### EEGNET ###########################
# taken from https://github.com/Amir-Hofo/EEGNet_Pytorch/blob/main/EEG-Net%20Pytorch.ipynb 
# made slight modifications to kernel sizes

class EEGNet(nn.Module):
    def __init__(self,
                 n_chans:int = 8,
                 n_time_samples:int = 100,
                 sampling_rate:int = 128,
                 F1:int = 16,
                 D:int = 4,
                 dropout_rate:float = 0.3):
        super(EEGNet, self).__init__()
        
        F2 = F1*D
        kernel_size_1= (1,round(n_time_samples/2)) 
        kernel_size_2= (n_chans, 1)
        kernel_size_3= (1, round(n_time_samples/8)) 
        kernel_size_4= (1, 1)
        
        kernel_avgpool_1= (1,2) 
        kernel_avgpool_2= (1,4) 
        
        ks0= int(round((kernel_size_1[0]-1)/2))
        ks1= int(round((kernel_size_1[1]-1)/2))
        kernel_padding_1= (ks0, ks1-1)
        ks0= int(round((kernel_size_3[0]-1)/2))
        ks1= int(round((kernel_size_3[1]-1)/2))
        kernel_padding_3= (ks0, ks1)
        
        # First Block
        self.conv1 = nn.Conv2d(1, F1, kernel_size_1, padding= kernel_padding_1)
        self.batch_norm1 = nn.BatchNorm2d(F1)
        self.depth_wise_conv = nn.Conv2d(F1, D*F1, kernel_size_2, groups=F1)
        self.batch_norm2 = nn.BatchNorm2d(D*F1)
        self.elu = nn.ELU()
        self.average_pool1 = nn.AvgPool2d(kernel_avgpool_1)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Second Block
        self.separable_conv_depth = nn.Conv2d(D*F1, D*F1, kernel_size_3, 
                                              padding= kernel_padding_3, groups= D*F1)
        self.separable_conv_point = nn.Conv2d(D*F1, F2, kernel_size_4)
        self.batch_norm3 = nn.BatchNorm2d(F2)
        self.average_pool2 = nn.AvgPool2d(kernel_avgpool_2)
        
        # Third Block
        self.flatten = nn.Flatten()
        # self.dense = nn.Sequential(nn.Linear(768, 256),
        #                          nn.ReLU(),
        #                          nn.Linear(256, 1))
        self.dense = nn.Linear(768, 1)
        
    def forward(self, x):

        out = self.batch_norm1(self.conv1(x))
        out = self.batch_norm2(self.depth_wise_conv(out))
        out = self.elu(out)
        out = self.dropout(self.average_pool1(out))
        
        out = self.separable_conv_point(self.separable_conv_depth(out))
        out= self.batch_norm3(out)
        out = self.elu(out)
        out = self.dropout(self.average_pool2(out))
        
        embeddings = self.flatten(out)
        return embeddings, self.dense(embeddings)  
    
    
class EEGNet_Original(nn.Module):
    def __init__(self,
                 n_chans:int = 8,
                 n_time_samples:int = 100,
                 sampling_rate:int = 128,
                 F1:int = 16,
                 D:int = 4,
                 dropout_rate:float = 0.5):
        super(EEGNet_Original, self).__init__()
        
        F2 = F1*D
        kernel_size_1= (1,round(sampling_rate/2)) 
        kernel_size_2= (n_chans, 1)
        kernel_size_3= (1, round(sampling_rate/8)) 
        kernel_size_4= (1, 1)
        
        kernel_avgpool_1= (1,4) 
        kernel_avgpool_2= (1,8) 
        dropout_rate= 0.2
        
        ks0= int(round((kernel_size_1[0]-1)/2))
        ks1= int(round((kernel_size_1[1]-1)/2))
        kernel_padding_1= (ks0, ks1-1)
        ks0= int(round((kernel_size_3[0]-1)/2))
        ks1= int(round((kernel_size_3[1]-1)/2))
        kernel_padding_3= (ks0, ks1)
        
        # layer 1
        self.conv2d = nn.Conv2d(1, F1, kernel_size_1, padding=kernel_padding_1)
        self.Batch_normalization_1 = nn.BatchNorm2d(F1)
        # layer 2
        self.Depthwise_conv2D = nn.Conv2d(F1, D*F1, kernel_size_2, groups= F1)
        self.Batch_normalization_2 = nn.BatchNorm2d(D*F1)
        self.Elu = nn.ELU()
        self.Average_pooling2D_1 = nn.AvgPool2d(kernel_avgpool_1)
        self.Dropout = nn.Dropout2d(dropout_rate)
        # layer 3
        self.Separable_conv2D_depth = nn.Conv2d( D*F1, D*F1, kernel_size_3,
                                                padding=kernel_padding_3, groups= D*F1)
        self.Separable_conv2D_point = nn.Conv2d(D*F1, F2, kernel_size_4)
        self.Batch_normalization_3 = nn.BatchNorm2d(F2)
        self.Average_pooling2D_2 = nn.AvgPool2d(kernel_avgpool_2)
        # layer 4
        self.Flatten = nn.Flatten()
        self.dense = nn.Linear(F2*round(n_time_samples/32), 1)
        
    def forward(self, x):
        # layer 1
        y = self.Batch_normalization_1(self.conv2d(x)) #.relu()
        # layer 2
        y = self.Batch_normalization_2(self.Depthwise_conv2D(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_1(y))
        # layer 3
        y = self.Separable_conv2D_depth(y)
        y = self.Batch_normalization_3(self.Separable_conv2D_point(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_2(y))
        # layer 4
        y = self.Flatten(y)
        
        return y, self.dense(y)
    