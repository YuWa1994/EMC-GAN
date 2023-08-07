# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 20:01:59 2022

@author: yuwa
"""
import torch
import torch.nn as nn

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Generator_1024V1(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=25, padding=12, stride=1),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=4, stride=4),
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, padding=3, stride=1),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=4, stride=4),      
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3, stride=1),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=4, stride=4),            
            )  

        self._fc = nn.Sequential(
            nn.Linear(in_features=512,out_features=1024), 
            nn.BatchNorm1d(1024),  
            #nn.LayerNorm(1024)
            )    
    def forward(self, x):    
        out = x.reshape(-1, 1, x.shape[1])
        out = self._conv(out)
        
        out = torch.flatten(out, 1)
        out = self._fc(out)
        return out  


##############################
#           U-NET
##############################
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv1d(in_size, out_size, kernel_size=7, padding=3, 
                            stride=1, bias=False),
                  nn.LeakyReLU(0.2),
                  nn.AvgPool1d(kernel_size=2, stride=2)]
        if normalize:
            layers.append(nn.InstanceNorm1d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
            
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose1d(in_size, out_size, kernel_size=7, 
                                     output_padding=1, padding=3, 
                                     stride=2, bias=False),
                  nn.LeakyReLU(0.2)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Generator_1024V2(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = UNetDown(1, 16, normalize=False)
        self.down2 = UNetDown(16, 32)
        self.down3 = UNetDown(32, 32)

        self.up1 = UNetUp(32, 32)
        self.up2 = UNetUp(64, 16)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(32, 1, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(1),  
        )

    def forward(self, X):
        X = X.reshape(X.shape[0], 1, -1)    
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(X)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        
        u1 = self.up1(d3, d2)
        u2 = self.up2(u1, d1)
        out = self.final(u2)
        return torch.flatten(out,1)
    
    
class Discriminator_1024V1(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.classes = classes
        self.DM = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=4, stride=4), 
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=15, padding=7, stride=1),
            nn.LeakyReLU(0.2),      
            nn.AvgPool1d(kernel_size=4, stride=4), 
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=15, padding=7, stride=1),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=2, stride=2), 
            #nn.BatchNorm1d(32),               
            )
        
        self.score = nn.Sequential(
            nn.Linear(1024, 1),
            )

        self.SM = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=4, stride=4), 
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=15, padding=7, stride=1),
            nn.LeakyReLU(0.2),      
            nn.AvgPool1d(kernel_size=4, stride=4), 
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=15, padding=7, stride=1),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=2, stride=2), 
            #nn.BatchNorm1d(32),              
            )

        self.FM = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(0.2),              
            nn.AvgPool1d(kernel_size=4, stride=4), 
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=15, padding=7, stride=1),
            nn.LeakyReLU(0.2),      
            nn.AvgPool1d(kernel_size=4, stride=4), 
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=15, padding=7, stride=1),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=2, stride=2), 
            #nn.BatchNorm1d(32),              
            )
        
        self.CM = nn.Sequential(
            nn.Linear(1024, classes),
            nn.Softmax(dim=1)
            )

    def _discriminator(self, X):    
        X = X.reshape(X.shape[0], 1, -1)      
        DM = self.DM(X)
        DM = torch.flatten(DM, 1)
        scores = self.score(DM)      
        return scores, DM  
        
    def _classifier(self, X1, DM):  
        X1 = X1.reshape(X1.shape[0], 1, -1)
        SM = self.SM(X1)        
        SM = SM.reshape(SM.shape[0], 1, -1) 
        
        DM = DM.reshape(DM.shape[0], 1, -1)
        out = torch.cat((SM, DM),1) 
        
        FM = self.FM(out) 
        label = self.CM(torch.flatten(FM, 1))               
        return label, torch.flatten(SM, 1), torch.flatten(FM, 1)    
    
    def forward(self, X):  
        scores, DM = self._discriminator(X)
        label, SM, FM = self._classifier(X, DM)
        return scores, label, DM, SM, FM     
    

class Discriminator_1024V2(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.classes = classes
        self.DM = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=10, stride=1, padding=5),
            nn.LeakyReLU(0.2),    
            nn.AvgPool1d(kernel_size=4, stride=4), 
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=4, stride=4), 
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=2, stride=2),  
            )
        
        self.score = nn.Sequential(
            nn.Linear(1024, 1),
            )

        self.SM1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),    
            nn.AvgPool1d(kernel_size=4, stride=4), 
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=4, stride=4), 
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=2, stride=2),  
            )

        self.SM2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=75, stride=1, padding=37),
            nn.LeakyReLU(0.2),    
            nn.AvgPool1d(kernel_size=4, stride=4), 
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=4, stride=4), 
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=2, stride=2),  
            )


        self.FM = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),    
            
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2),                
            nn.AvgPool1d(kernel_size=4, stride=4), 
            
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=4, stride=4), 
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=2, stride=2),  
            )
        
        self.CM = nn.Sequential(
            nn.Linear(1024, classes),
            nn.Softmax(dim=1)
            )
        
        self.activ = nn.Sigmoid()

    def _discriminator(self, X):    
        X = X.reshape(X.shape[0], 1, -1)      
        DM = self.DM(X)
        DM = torch.flatten(DM, 1)
        scores = self.score(DM)      
        return scores, DM  
        
    def _classifier(self, X1, DM):  
        X1 = X1.reshape(X1.shape[0], 1, -1)
        SM1 = self.SM1(X1)        
        SM1 = SM1.reshape(SM1.shape[0], 1, -1) 

        SM2 = self.SM2(X1)        
        SM2 = SM1.reshape(SM2.shape[0], 1, -1) 
        
        DM = DM.reshape(DM.shape[0], 1, -1)
        out = torch.cat((SM1, SM2, DM),1) 
        #out = self.activ(SM+DM)
        
        FM = self.FM(out) 
        
        label = self.CM(torch.flatten(FM, 1))               
        return label, torch.flatten(SM1, 1), torch.flatten(FM, 1)

    def forward(self, X):  
        scores, DM = self._discriminator(X)
        label, SM, FM = self._classifier(X, DM)
        return scores, label, DM, SM, FM 



class Generator_8192(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=10, padding=5, stride=1),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=4, stride=4),
            #nn.BatchNorm1d(8),  
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=25, padding=12, stride=1),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=4, stride=4),
            #nn.BatchNorm1d(16),         
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=25, padding=12, stride=1),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=4, stride=4),
            #nn.BatchNorm1d(32),               
            )  

        self._fc = nn.Sequential(
            nn.Linear(in_features=4096,out_features=8192), 
            )    
    def forward(self, x):    
        out = x.reshape(-1, 1, 8192)
        out = self._conv(out)
        
        out = torch.flatten(out, 1)
        out = self._fc(out)
        return out  


class Discriminator_8192(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.classes = classes
        self.DM = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=10, stride=1, padding=5),
            nn.LeakyReLU(0.2),    
            nn.AvgPool1d(kernel_size=4, stride=4), 
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=25, stride=1, padding=12),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=4, stride=4), 
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=25, stride=1, padding=12),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=2, stride=2),  
            )
        
        self.score = nn.Sequential(
            nn.Linear(8192, 1),
            )

        self.SM = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=10, stride=1, padding=5),
            nn.LeakyReLU(0.2),    
            nn.AvgPool1d(kernel_size=4, stride=4), 
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=25, stride=1, padding=12),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=4, stride=4), 
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=25, stride=1, padding=12),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=2, stride=2),  
            )

        self.FM = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=10, stride=1, padding=5),
            nn.LeakyReLU(0.2),    
            nn.AvgPool1d(kernel_size=4, stride=4), 
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=25, stride=1, padding=12),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=4, stride=4), 
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=25, stride=1, padding=12),
            nn.LeakyReLU(0.2),  
            nn.AvgPool1d(kernel_size=2, stride=2),  
            )
        
        self.CM = nn.Sequential(
            nn.Linear(8192, classes),
            nn.Softmax(dim=1)
            )

    def _discriminator(self, X):    
        X = X.reshape(X.shape[0], 1, -1)      
        DM = self.DM(X)
        DM = torch.flatten(DM, 1)
        scores = self.score(DM)      
        return scores, DM  
        
    def _classifier(self, X1, DM):  
        X1 = X1.reshape(X1.shape[0], 1, -1)
        SM = self.SM(X1)        
        SM = SM.reshape(SM.shape[0], 1, -1) 
        
        DM = DM.reshape(DM.shape[0], 1, -1)
        out = torch.cat((SM, DM),1) 
        #out = out + torch.cat((X1, X1),1)  # residual
        
        FM = self.FM(out) 
        
        label = self.CM(torch.flatten(FM, 1))               
        return label, torch.flatten(SM, 1), torch.flatten(FM, 1)
        return label, torch.flatten(SM, 1), torch.flatten(FM, 1)

    def forward(self, X):  
        scores, DM = self._discriminator(X)
        label, SM, FM = self._classifier(X, DM)
        return scores, label, DM, SM, FM 
    
    
    
