# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:39:14 2022

@author: yuwa
"""

import numpy as np
import torch
import YW_utils as utils
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import os
import GAN_Nets


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class GAN_for_ENC():
    ''' GAN for emerging new classes problem
    
    Parameters
    ----------
    X : 
        Array-like of shape (n_samples, n_features),
    label : 
        Array-like of shape (n_samples, 1),    
    buffer_size : 
        Number of samples in buffer
        
    Returns
    -------        
    Prediction
    '''    
    def __init__(self, X,
                 label=None, 
                 n_epoch=100, batch_size=100, sample_interval=1,
                 learn_rate=0.001,
                 alpha=0.5,
                 savepath=None,
                 buffer_size=250,
                 noise=False,
                 is_normalization=True, check_loss=True,
                 train_method='WGAN-gp',
                 ): 
        super().__init__()
        
        self.X = X     
        if is_normalization:
            self.trainX, self.zscore_mu, self.zscore_sigma = utils.zscore(self.X, dim=0)

        self.label = label
        
        self.learn_rate = learn_rate
        self.n_epoch = n_epoch
        self.batch_size = batch_size  
        self.sample_interval = sample_interval
        self.alpha = alpha
        self.savepath = savepath
        self.buffer = []
        self.buffer_size = buffer_size    
        self.check_loss = check_loss
        self.count = 0
        
        self.noise = noise
        self.is_normalization = is_normalization
        self.train_method = train_method

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_num = X.shape[0]

        # For multi-classification
        self.loss_func = nn.MSELoss() 
        if self.label is None: 
            self.label = np.zeros(self.X.shape[0])
        else:   
            self.label = self.label
        self.fake_label = np.max(self.label)+1                                
        self.n_classes = len(np.unique(self.label))+1 
        self.label = F.one_hot(torch.from_numpy(self.label).long(), num_classes=self.n_classes)   
        self.label = torch.flatten(self.label, 1)
        
        if self.savepath is None:
            self.savepath = './history'
            
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        utils.empty_file(self.savepath)

        self.returns = {}
        self.returns['history_states_idx'] = []
        self.returns['gen_X'] = []  # save generated data during training
        self.returns['gen_X_label'] = []
        self.returns['gen_X_scores'] = []  # save the scores of generated data during training
        self.returns['loss_D'] = []
        self.returns['loss_G'] = []
        self.returns['loss_C'] = []
        self.returns['loss_D_his'] = []
        self.returns['loss_G_his'] = []
        self.returns['loss_C_his'] = []
                
    def train(self, trainX=None, G=None, D=None, G_init=None, D_init=None,
              learn_rate=None, n_epoch=None): 
        
        # -----------------
        #  Setting
        # -----------------                
        if trainX is None: trainX = self.trainX
        if G_init is not None: G.load_state_dict(torch.load(G_init)['model_state_dict'])
        if D_init is not None: G.load_state_dict(torch.load(D_init)['model_state_dict'])
        if learn_rate is None: learn_rate = self.learn_rate
        if n_epoch is None: n_epoch = self.n_epoch
        trainX = torch.from_numpy(trainX).to(self.device)
        generator = G.to(self.device)
        discriminator = D.to(self.device)

        # -----------------
        #  Train GAN
        # -----------------            
        self.global_iter = 0
        for epoch in range(n_epoch):  
            indices = torch.randperm(trainX.shape[0])
            #indices = torch.from_numpy(np.arange(trainX.shape[0])).type(torch.long)
            indices = torch.split(indices, self.batch_size)  
            
            #if  epoch>=100:
            #    for g in optimizer_D.param_groups: g['lr'] = g['lr']*(1-0.001)     
            #    for g in optimizer_G.param_groups: g['lr'] = g['lr']*(1-0.001)  
            
            batch_size = len(indices)
            if self.train_method == 'WGAN-gp':
                self.train_WGAN_gp(epoch, generator, discriminator, trainX, indices, batch_size, learn_rate)           
            elif self.train_method == 'WGAN':
                self.train_WGAN(epoch, generator, discriminator, trainX, indices, batch_size)   
            elif self.train_method == 'WGAN-cls':
                self.train_WGAN_cls(epoch, generator, discriminator, trainX, indices, batch_size)                  
            elif self.train_method == 'WGAN-div':
                self.train_WGAN_div(epoch, generator, discriminator, trainX, indices, batch_size)                  
                
                
            # Save generated data during training
            if (epoch+1) % self.sample_interval == 0:
                utils.save_checkpoint(discriminator, self.savepath+'/D_'+str(epoch+1), epoch+1)
                #utils.save_checkpoint(generator, self.savepath+'/G_'+str(epoch+1), epoch+1)  
                gen_X = generator(self.z)
                out = discriminator(gen_X)
                self.returns['gen_X'].append(gen_X.cpu().detach().numpy())
                self.returns['gen_X_scores'].append(out[0].cpu().detach().numpy())                 
            
        utils.save_checkpoint(discriminator, self.savepath+'/D_'+str(epoch+1), epoch+1) 
        return self

    
    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1,)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        with torch.backends.cudnn.flags(enabled=False):
            out = D(interpolates)
        d_interpolates = out[0]
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=fake,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
        #gradient_penalty = ((gradients.norm(2, dim=1) - 0) ** 2).mean()
        return gradient_penalty

    def Compute_W_div_gradient_penalty(self, D, real_X, real_validity, fake_X, fake_validity, k, p):
        # Compute W-div gradient penalty
        real_grad_out = Variable(Tensor(real_X.size(0), 1).fill_(1.0), requires_grad=False)
        real_grad = autograd.grad(outputs=real_validity, 
                                  inputs=real_X, 
                                  grad_outputs=real_grad_out, 
                                  create_graph=True, 
                                  retain_graph=True, 
                                  only_inputs=True)[0]
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

        fake_grad_out = Variable(Tensor(fake_X.size(0), 1).fill_(1.0), requires_grad=False)
        fake_grad = autograd.grad(outputs=fake_validity, 
                                  inputs=fake_X, 
                                  grad_outputs=fake_grad_out, 
                                  create_graph=True, 
                                  retain_graph=True, 
                                  only_inputs=True)[0]
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2   
        return div_gp
        

    def train_WGAN_gp(self, epoch, generator, discriminator, trainX, indices, batch_size,
                      learn_rate=None, n_critic_g=10, lambda_gp=2):
        criterion_pixelwise = torch.nn.L1Loss()
        if epoch==0:
            self.n_critic_g, self.lambda_gp = n_critic_g, lambda_gp
            
            self.loss_func = nn.MSELoss() 
            self.optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=learn_rate) 
            self.optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=learn_rate)   
        
        for i in range(batch_size):  
            # Configure input     
            real_X = Variable(trainX[indices[i]]).float()
            true_label = self.label[indices[i]]

            if self.noise: # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (real_X.shape[0], int(real_X.shape[1]/8)))))            
            else: # real_imgs as generator input
                z = real_X
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.z = z
            fake_X = generator(z).detach()
            with torch.backends.cudnn.flags(enabled=False):
                out = discriminator(real_X)
            real_validity = out[0]
            real_predict = out[1]
            
            with torch.backends.cudnn.flags(enabled=False):
                out = discriminator(fake_X)
            fake_validity = out[0]
            fake_predict = out[1]
            gradient_penalty = self.compute_gradient_penalty(discriminator, 
                                                             real_X.data, fake_X.data)
            # Adversarial loss
            loss_D = (torch.mean(fake_validity) - torch.mean(real_validity) 
                      + self.lambda_gp * gradient_penalty)    
            
            # -----------------------------
            #  Train Multi-classification
            # -----------------------------                
            predict_label = torch.cat((real_predict, fake_predict))
            fake_label = F.one_hot(self.fake_label*torch.ones(true_label.shape[0]).long())
            batch_label = torch.cat((true_label, fake_label)).to(self.device)
            loss_C = self.loss_func(predict_label, batch_label.to(torch.float32))
            
            loss = self.alpha * loss_D + (1-self.alpha) * loss_C             

            self.optimizer_D.zero_grad()
            loss.backward()
            self.optimizer_D.step()                

            # -----------------
            #  Train Generator
            # -----------------
            # Train the generator every n_critic steps
            if self.global_iter % self.n_critic_g == 0:
                fake_X = generator(z)
                loss_pixel = criterion_pixelwise(fake_X, z)
                
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                out = discriminator(fake_X)
                fake_validity = out[0]
                loss_G = -torch.mean(fake_validity) + 100*loss_pixel
                
                self.optimizer_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()     
            self.global_iter += 1
         
        if self.check_loss:
            loss_G, loss_D, loss_C = self.check_loss_fun(trainX, generator, 
                                                 discriminator, self.loss_func)           
            self.returns['loss_D'].append(loss_D)
            self.returns['loss_G'].append(loss_G)
            self.returns['loss_C'].append(loss_C)    
            print(
                "[Epoch %d/%d] [D : %f] [G : %f] [C : %f]"
                % (epoch, self.n_epoch, loss_D, loss_G, loss_C))


    def train_WGAN_cls(self, epoch, generator, discriminator, trainX, indices, batch_size,
                      n_critic_g=3, lambda_gp=2):
        if epoch==0:
            self.n_critic_g, self.lambda_gp = n_critic_g, lambda_gp
            
            self.loss_func = nn.MSELoss() 
            self.pixelwise_loss = torch.nn.L1Loss()
            self.optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.0001) 
            self.optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=0.0001)   
        
        for i in range(batch_size):  
            # Configure input     
            real_X = Variable(trainX[indices[i]]).float()
            true_label = self.label[indices[i]]

            if self.noise: # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (real_X.shape[0], int(real_X.shape[1]/8)))))            
            else: # real_imgs as generator input
                z = real_X
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            fake_X = generator(z).detach()
            with torch.backends.cudnn.flags(enabled=False):
                out = discriminator(real_X)
            real_validity = out[0]
            real_predict = out[1]
            
            with torch.backends.cudnn.flags(enabled=False):
                out = discriminator(fake_X)
            fake_validity = out[0]
            fake_predict = out[1]
            gradient_penalty = self.compute_gradient_penalty(discriminator, 
                                                             real_X.data, fake_X.data)
            # Adversarial loss
            loss_D = (torch.mean(fake_validity) - torch.mean(real_validity) 
                      + self.lambda_gp * gradient_penalty)    
            
            # -----------------------------
            #  Train Multi-classification
            # -----------------------------                
            predict_label = torch.cat((real_predict, fake_predict))
            fake_label = F.one_hot(self.fake_label*torch.ones(true_label.shape[0]).long())
            batch_label = torch.cat((true_label, fake_label)).to(self.device)
            loss_C = self.loss_func(predict_label, batch_label.to(torch.float32))
            
            loss = self.alpha * loss_D + (1-self.alpha) * loss_C             

            self.optimizer_D.zero_grad()
            loss.backward()
            self.optimizer_D.step()                

            # -----------------
            #  Train Generator
            # -----------------
            # Train the generator every n_critic steps
            if i % self.n_critic_g == 0:
                fake_X = generator(z)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                out = discriminator(fake_X)
                fake_validity = out[0]
                fake_predict = out[1]
                loss_G = -torch.mean(fake_validity) 
                #loss_G += 0.5*(self.pixelwise_loss(fake_X, z))
                
                self.optimizer_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()        
         
        if self.check_loss:
            loss_G, loss_D, loss_C = self.check_loss_fun(trainX, generator, 
                                                 discriminator, self.loss_func)           
            self.returns['loss_D'].append(loss_D)
            self.returns['loss_G'].append(loss_G)
            self.returns['loss_C'].append(loss_C)    
            print(
                "[Epoch %d/%d] [D : %f] [G : %f] [C : %f]"
                % (epoch, self.n_epoch, loss_D, loss_G, loss_C))
            
                
    def train_WGAN(self, epoch, generator, discriminator, trainX, indices, batch_size,
                        clip_value=0.01):
        if epoch==0:
            self.clip_value = clip_value
            
            self.loss_func = nn.MSELoss() 
            self.optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.0001) 
            self.optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=0.0001)         
        
        for i in range(batch_size):  
            # Configure input
            real_X = Variable(trainX[indices[i]]).float()
            true_label = self.label[indices[i]]

            if self.noise: # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (real_X.shape[0], int(real_X.shape[1]/8)))))            
            else: # real_imgs as generator input
                z = real_X
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            fake_X = generator(z).detach()
            with torch.backends.cudnn.flags(enabled=False):
                out = discriminator(real_X)
            real_validity = out[0]
            real_predict = out[1]
            
            with torch.backends.cudnn.flags(enabled=False):
                out = discriminator(fake_X)
            fake_validity = out[0]
            fake_predict = out[1]

            # Adversarial loss
            loss_D = torch.mean(fake_validity) - torch.mean(real_validity) 
            
            # -----------------------------
            #  Train Multi-classification
            # -----------------------------                
            predict_label = torch.cat((real_predict, fake_predict))
            fake_label = F.one_hot(self.fake_label*torch.ones(true_label.shape[0]).long())
            batch_label = torch.cat((true_label, fake_label)).to(self.device)
            loss_C = self.loss_func(predict_label, batch_label.to(torch.float32))
            
            loss = self.alpha * loss_D + (1-self.alpha) * loss_C             
            
            self.optimizer_D.zero_grad()
            loss.backward()
            self.optimizer_D.step()                

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)

            # -----------------
            #  Train Generator
            # -----------------
            # Train the generator every n_critic steps
            if i % self.n_critic_g == 0:
                fake_X = generator(z)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                out = discriminator(fake_X)
                fake_validity = out[0]
                loss_G = -torch.mean(fake_validity)
    
                self.optimizer_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()        
                
        if self.check_loss:
            loss_G, loss_D, loss_C = self.check_loss_fun(trainX, generator, 
                                                 discriminator, self.loss_func)           
            self.returns['loss_D'].append(loss_D)
            self.returns['loss_G'].append(loss_G)
            self.returns['loss_C'].append(loss_C)    
            print(
                "[Epoch %d/%d] [D : %f] [G : %f] [C : %f]"
                % (epoch, self.n_epoch, loss_D, loss_G, loss_C))        


    def train_WGAN_div(self, epoch, generator, discriminator, trainX, indices, batch_size,
                       n_critic_g=3, k=2, p=2):
        if epoch==0:
            self. n_critic_g, self.k, self.p = n_critic_g, k, p
            
            self.loss_func = nn.MSELoss() 
            self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
            self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))           
        
        for i in range(batch_size):  
            # Configure input
            real_X = Variable(trainX[indices[i]]).float()
            true_label = self.label[indices[i]]
            
            if self.noise: # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (real_X.shape[0], int(real_X.shape[1]/8)))))           
            else: # real_imgs as generator input
                z = real_X                   
            # ---------------------
            #  Train Discriminator
            # ---------------------
            with torch.backends.cudnn.flags(enabled=True):
                real_X = real_X.requires_grad_(True)
                out = discriminator(real_X)
            real_validity = out[0]
            real_predict = out[1]
            
            with torch.backends.cudnn.flags(enabled=True):
                fake_X = generator(z).detach()
                fake_X = fake_X.requires_grad_(True)
                out = discriminator(fake_X)
            fake_validity = out[0]
            fake_predict = out[1]
            
            div_gp = self.Compute_W_div_gradient_penalty(discriminator, real_X, real_validity,
                                                         fake_X,fake_validity, k, p)
            
            # Adversarial loss
            loss_D = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp
            
            # -----------------------------
            #  Train Multi-classification
            # -----------------------------                
            predict_label = torch.cat((real_predict, fake_predict))
            fake_label = F.one_hot(self.fake_label*torch.ones(true_label.shape[0]).long())
            #fake_label = true_label
            batch_label = torch.cat((true_label, fake_label)).to(self.device)
            loss_C = self.loss_func(predict_label, batch_label.to(torch.float32))
            
            loss = self.alpha * loss_D + (1-self.alpha) * loss_C             
            
            self.optimizer_D.zero_grad()
            loss.backward()
            self.optimizer_D.step()                

            # -----------------
            #  Train Generator
            # -----------------                 
            # Train the generator every n_critic steps
            if i % self.n_critic_g == 0:
                fake_X = generator(z)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                out = discriminator(fake_X)
                fake_validity = out[0]
                loss_G = -torch.mean(fake_validity)
    
                self.optimizer_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()        
                
        if self.check_loss:
            loss_G, loss_D, loss_C = self.check_loss_fun(trainX, generator, 
                                                 discriminator, self.loss_func)           
            self.returns['loss_D'].append(loss_D)
            self.returns['loss_G'].append(loss_G)
            self.returns['loss_C'].append(loss_C)    
            print(
                "[Epoch %d/%d] [D : %f] [G : %f] [C : %f]"
                % (epoch, self.n_epoch, loss_D, loss_G, loss_C))     
            

    def check_loss_fun(self, trainX, generator, discriminator, loss_func):
        loss_G = 0
        loss_D = 0
        loss_C = 0
        
        indices = torch.from_numpy(np.arange(trainX.shape[0])).type(torch.long)
        indices = torch.split(indices, self.batch_size)           
        for i in range(len(indices)):  
            # Configure input
            real_X = Variable(trainX[indices[i]]).float()
            true_label = self.label[indices[i]]

            if self.noise: # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (real_X.shape[0], int(real_X.shape[1]/8)))))     
            else: # real_imgs as generator input
                z = real_X
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            fake_X = generator(z).detach()
            
            out = discriminator(real_X)
            real_validity = out[0]
            real_predict = out[1]
            
            out = discriminator(fake_X)
            fake_validity = out[0]
            fake_predict = out[1]
            
            if self.train_method == 'WGAN':
                loss_D = torch.mean(fake_validity) - torch.mean(real_validity) 
                loss_G += -torch.mean(fake_validity).item()
            elif self.train_method == 'WGAN-gp':
                gradient_penalty = self.compute_gradient_penalty(discriminator, 
                                                             real_X.data, fake_X.data)
                loss_D += (torch.mean(fake_validity) - torch.mean(real_validity) 
                          + self.lambda_gp * gradient_penalty).item()            
                loss_G += -torch.mean(fake_validity).item()
                
            elif self.train_method == 'WGAN-cls':
                gradient_penalty = self.compute_gradient_penalty(discriminator, 
                                                             real_X.data, fake_X.data)
                loss_D += (torch.mean(fake_validity) - torch.mean(real_validity) 
                          + self.lambda_gp * gradient_penalty).item()       
                loss_G += -torch.mean(fake_validity).item() + loss_func(fake_X, z).item()
                
            elif self.train_method == 'WGAN-div':
                div_gp = self.Compute_W_div_gradient_penalty(discriminator, real_X, real_validity,
                                                             fake_X,fake_validity, self.k, self.p)
                loss_D = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp
                loss_G += -torch.mean(fake_validity).item()
                
            
            # -----------------------------
            #  Train Multi-classification
            # -----------------------------                
            predict_label = torch.cat((real_predict, fake_predict))
            fake_label = F.one_hot(self.fake_label*torch.ones(true_label.shape[0]).long())

            batch_label = torch.cat((true_label, fake_label)).to(self.device)
            loss_C += loss_func(predict_label, batch_label.to(torch.float32)).item()
            
        loss_G = loss_G/len(indices)
        loss_D = loss_D/len(indices)
        loss_C = loss_C/len(indices)
        return loss_G, loss_D, loss_C

    
    def get_scores(self, X, D=None, state_dict=None):
        if self.is_normalization: 
            X = utils.normalize(X, self.zscore_mu, self.zscore_sigma)   
             
        D.load_state_dict(torch.load(state_dict)['model_state_dict'])
        D.eval()
            
        X = torch.from_numpy(X).to(self.device)
        X = torch.split(X, 50)
        score = []
        pred_label = []
        DM = []
        SM = []
        FM = []
        attn = []
        attn2 = []
        attn3 = []
        for i in range(len(X)):
            out  = D(X[i])   
            score.extend(out[0].detach().cpu().numpy())
            pred_label.extend(out[1].detach().cpu().numpy())
            #DM.extend(out[2].detach().cpu().numpy())
            #SM.extend(out[3].detach().cpu().numpy())
            #FM.extend(out[4].detach().cpu().numpy())
            if len(out)==6:
                attn.extend(out[5].detach().cpu().numpy())
            if len(out)==7:
               attn.extend(out[5].detach().cpu().numpy())     
               attn2.extend(out[6].detach().cpu().numpy())
            if len(out)==8:
               attn.extend(out[5].detach().cpu().numpy())     
               attn2.extend(out[6].detach().cpu().numpy()) 
               attn3.extend(out[6].detach().cpu().numpy())
        return score, pred_label, DM, SM, FM, attn, attn2
    
    
    def get_scores_update(self, X, D=None, G=None, update=False, 
                          warm_up=100, basemodel_num=30, fix_model=-1, stride=3):          
        D.eval(), G.eval()
        self.get_ensemble_models(warm_up, basemodel_num, stride, fix_model)
        #self.get_ensemble_models2(warm_up, basemodel_num, stride)
        
        num = X.shape[0]
        X = np.array_split(X, num)
        score = []
        pred_prabability = []
        prediction = []
        DM, SM, FM = [], [], []
        self.update_time = []
        self.buffer_prob = []
        update_times =  0
        for i in range(num):
            data = X[i]
            if self.is_normalization: 
                data = utils.normalize(data, self.zscore_mu, self.zscore_sigma)               
            data = torch.from_numpy(data).to(self.device)
            
            s, p = 0, 0
            for m, epoch in enumerate(self.ensemble_models):
                D.load_state_dict(torch.load(f'{self.savepath}/D_{epoch+1}')['model_state_dict'])
                out = D(data)   
                s += out[0].detach().cpu().numpy()
                p += out[1].detach().cpu().numpy()
                
                if m==0:
                    DM.extend(out[2].detach().cpu().numpy())
                    SM.extend(out[3].detach().cpu().numpy())
                    FM.extend(out[4].detach().cpu().numpy())                
                
            score.extend(s/len(self.ensemble_models))
            pred_prabability.extend(p/len(self.ensemble_models))
            prediction.extend(np.argmax(p, axis=1))
            
            if update:
                # replace the training data
                if pred_prabability[i][prediction[i]]>1:
                    if prediction[i]<self.n_classes-1 and np.random.uniform(size=1)>0.8:
                        out = np.where(np.argmax(self.label.numpy(), axis=1)==prediction[i])
                        self.X[np.random.randint(len(out[0]), size=1)]=X[i]          
                
                if prediction[i]>self.n_classes-2:
                    #if i>=2 and (prediction[i]==prediction[i-1]==prediction[i-2]):
                    #if pred_prabability[i][prediction[i]]>0.6:
                    self.buffer.extend(X[i])
                    self.buffer_prob.append(pred_prabability[i])
                    if np.array(self.buffer).shape[0]>=self.buffer_size:
                        self.update_time.append(i)
                        print(f'-----Update model: {len(self.update_time)}, time {i}----')
                        G, D = self.update_model(G, D)
                        self.get_ensemble_models(warm_up, basemodel_num, stride, fix_model)
                        update_times += 1
            print(f'Process stream instance: {i}, prediction {prediction[i]}, updated {update_times} time, buffer size {len(self.buffer)}')
        return score, pred_prabability, prediction, DM, SM, FM


    def update_model(self, G, D):
        # save the old model to history file
        save_old_mode = f'{self.savepath}/old_model'
        if not os.path.exists(save_old_mode): os.makedirs(save_old_mode)    
        utils.save_checkpoint(G, f'{save_old_mode}/G_class_'+str(self.n_classes))
        utils.save_checkpoint(D, f'{save_old_mode}/D_class_'+str(self.n_classes))
        
        # Filter buffer
        E = []
        for i, n in enumerate(self.buffer_prob):
            E.append(np.sum(-1*n * np.log(n+1E-8)))
        th = np.mean(E) + 3*np.std(E)   # 3
        
        # update training set and label
        for i, data in enumerate(self.buffer):
            if i in np.where(np.array(E)<th)[0]:
                self.X = np.append(self.X, data.reshape(1,-1), axis=0)    
        
        if self.is_normalization:
            self.trainX, self.zscore_mu, self.zscore_sigma = utils.zscore(self.X, dim=0)        
        
        self.label = np.append(np.argmax(self.label.numpy(), axis=1), 
                      np.ones(len(np.where(np.array(E)<th)[0]))*self.n_classes-1, 
                      axis=0)
        self.fake_label = int(np.max(self.label)+1)                               
        self.n_classes = self.n_classes+1 
        self.label = F.one_hot(torch.from_numpy(self.label).long(), num_classes=self.n_classes)   
        self.label = torch.flatten(self.label, 1)
        self.buffer, self.buffer_prob = [], []
        
        self.returns = {}
        self.returns['history_states_idx'] = []
        self.returns['gen_X'] = []  # save generated data during training
        self.returns['gen_X_label'] = []
        self.returns['gen_X_scores'] = []  # save the scores of generated data during training
        self.returns['loss_D'] = []
        self.returns['loss_G'] = []
        self.returns['loss_C'], self.returns['loss'] =[], []
        
        # set network
        utils.setup_seed(seed=0)
        G_new = GAN_Nets.Generator_1024V1().to(self.device)
        D_new = GAN_Nets.Discriminator_1024V2(classes=self.n_classes).to(self.device)
        
        # train the new model   
        self = self.train(self.trainX, G_new, D_new, 
                          learn_rate=self.learn_rate, n_epoch=self.n_epoch)        
        return G.eval(), D_new.eval()
    

    def get_ensemble_models(self, warm_up, basemodel_num, stride=1, fix_model=-1):
        self.ensemble_models = []
        if fix_model>-1:
            self.ensemble_models = fix_model
        else:
            self.warm_up = warm_up
            self.basemodel_num = basemodel_num
            loss_C = self.returns['loss_C']
            idx = np.arange(warm_up,len(loss_C))[::stride] 

            sorted_id = sorted(range(len(loss_C)), key=lambda k: loss_C[k], reverse=False)
            self.ensemble_models = [i for i in sorted_id if i in idx]



def load_my_state_dict(model, pre_train):
    pretrain_dict = pre_train.state_dict()
    model_dict = model.state_dict()
    
    for k, v in pretrain_dict.items():
        if model_dict[k].shape[0]==pretrain_dict[k].shape[0]:
            model_dict[k] = pretrain_dict[k]
        else:
            model_dict[k][:-1] = pretrain_dict[k]
    
    #model_dict.update(pretrain_dict) 
    model.load_state_dict(model_dict)
    return model    

def load_my_state_dict2(model, pre_train, alpha=0.5):
    pretrain_dict = pre_train.state_dict()
    model_dict = model.state_dict()
    
    for k, v in pretrain_dict.items():
        if model_dict[k].shape==pretrain_dict[k].shape:
            model_dict[k] = model_dict[k] - alpha*pretrain_dict[k]
        else:
            model_dict[k][:-1] = model_dict[k][:-1] - alpha*pretrain_dict[k]
    model.load_state_dict(model_dict)
    return model    

        
def Mode(L, method='max'):
    # find the most frequent element in a list
    all_count = []
    for i, n in enumerate(np.unique(L)): all_count.append(L.count(n))
    if method == 'max': 
        count = np.max(all_count)
        count_idx = np.unique(L)[np.where(all_count == count)[0]][0]
    return count_idx, count

def KL_loss(hn, sparsityTarget=0.001):    
    KL = sparsityTarget * torch.log(sparsityTarget/hn) + (1-sparsityTarget) * torch.log((1-sparsityTarget) / (1-hn+1e-8))  
    return KL.sum()


def update_weight(pretrained_dict, model):
    model_dict = model.state_dict()
    # Filter out unnecessary keys
    classifier_W = pretrained_dict.state_dict()['classifier.0.weight']
    classifier_B = pretrained_dict.state_dict()['classifier.0.bias']    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 

    model_dict.update(pretrained_dict)    
    model_dict['classifier.0.weight'] = classifier_W
    model_dict['classifier.0.bias'] = classifier_B
    model.load_state_dict(model_dict)
    return model
