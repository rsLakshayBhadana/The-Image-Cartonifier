#!/usr/bin/env python
# coding: utf-8

# # Neural Style Transfer 

# In[202]:


import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
import copy 
from PIL import Image 
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms 
import torchvision.models as models 


# In[203]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device 
content_layer_default =["conv_4"]
style_layers_default = ["conv_1","conv_2","conv_3","conv_4","conv_5"]
num_steps = 300


# # Importing the VGG16 model 

# In[204]:


cnn = models.vgg19(pretrained = True).features.to(device).eval()


# In[205]:


cnn_normalization_mean = torch.tensor([0.485,0.456,0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229,0.224,0.225]).to(device)


# In[206]:


# Converting Image to required size and to tensor 
def image_loader(image_name,imgsize):
    image = Image.open(image_name)
    # Scaling it and converting it into tensor 
    transform = transforms.Compose([transforms.Resize(imgsize),transforms.CenterCrop(imgsize), transforms.ToTensor()])
    # Reshaping the input data to fit into the network's input dimensions(fake batch dimension)
    image = transform(image).unsqueeze(0)
    return image.to(device , torch.float)


# **Showing the images using PIL of tensor type**

# In[207]:


def imshow(tensor,title=None):
    image = tensor.cpu().clone() # Copyiny it so any change on image not effect tensor 
    image  = image.squeeze(0)   # Removing the fake batch dimension 
    # Converting the image to PIL form 
    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)      # pause a bit so that plots are updated 


# # Calculating the Content Loss 

# In[208]:


class ContentLoss(nn.Module):
    def __init__(self,target,):
        super(ContentLoss,self).__init__()
        # Detaching the traget Content (So not show error ?)
        self.target = target.detach()
        
    def forward(self,input):
        # here input is the image that we are iteratively changing 
        self.loss  = F.mse_loss(input,self.target)
        return input 


# ## Calculating the Gram matrix 

# In[209]:


def gram_matrix(input):
    a,b,c,d = input.size()
    # a = batch size 
    # b = no. of feature maps 
    # (c,d) = dimensions of a previous feature map (N=c*d)
    
    features  = input.view(a*b,c*d) 
    G = torch.mm(features,features.t())  # gram product 
    
    # Normalising 
    return G.div(a*b*c*d)


# ## Style Loss

# In[210]:


class StyleLoss(nn.Module):
    
    def __init__(self,target_features):
        super(StyleLoss,self).__init__()
        # Detaching the traget Content (So not show error ?)
        self.target = gram_matrix(target_feature).detach()
        
    def forward(self,input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G,self.target)
        return input


# ## Normalization 

# In[211]:


# Helper for VGG16 
class Normalization(nn.Module):
    def __init__(self,mean,std):
        super(Normalization,self).__init__()
        # Viewing the mean and std to make them[c*1*1] so that they  can be used directly for work with 
        # image tensor of shape [b*c*h*w] ,b=batch suze , c = no. of channels , h=heigt & w=width
        self.mean= torch.tensor(mean).view(-1,1,1)
        self.std  = torch.tensor(std).view(-1,1,1)
        
    def forward(self,img):
        # normalize it (img)
        return (img-self.mean)/self.std


# ### optimiser 

# In[212]:


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer 


# ## Changes in the VGG model and storing the losses from both the contant and style

# In[213]:


def get_style_model_losses(cnn,normalization_mean,normalization_std,style_img,content_img,content_layers=content_layer_default,
                           style_layers=style_layers_default):
    # Not change it 
    cnn = copy.deepcopy(cnn)
    
    #normalisation
    normalization = Normalization(normalization_mean,normalization_std).to(device)
    
    # Losses (cretaing the list )
    content_losses =[]
    style_losses = []
    
    # Assuming that cnn is a nn.sequential, so we make a new nn.sequential to put in module 
    # that are suposed to be activated sequentially 
    model = nn.Sequential(normalization)
    
    i = 0
    # Iterating through every layer of cnn
    for layer in cnn.children():
        if isinstance(layer,nn.Conv2d):
            i+=1 
            name = 'conv_{}'.format(i)
        elif isinstance(layer,nn.ReLU):
            name = 'relu_{}'.format(i)
            # in_place version not work nicely for ContentLossand styleLoss so we replace it with out_place ones here
            
            layer = nn.ReLU(inplace = False)
            
        elif isinstance(layer,nn.MaxPool2d):
            name = 'pool_{}'.format(i)

        elif isinstance(layer,nn.Batchnorm2d):
            name = 'bn_{}'.format(i)
        
        else :
            raise RuntimeError('unknown layer: {}'.format(layer.__class__.__name__))
            
        model.add_module(name,layer)
        
        if name is content_layers:
            # add content loss 
            target = model(contnt_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i),content_loss)
            content_losses.append(content_loss)
            
        if name is style_layers:
            # add the style loss
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i),style_loss)
            style_losses.append(style_loss)
            
        # Detaching the last layers 
        for i in range(len(model)-1,-1,-1):
            if isinstance(model[i],ContentLoss) or isinstance(model[i],StyleLoss):
                break
                
        model = model[:(i+1)]
        
        return model , style_losses , content_losses


# In[217]:


# Taking the white noise image 
input_img = torch.randn(content_img.data.size(),device = device )


# # Final Model 

# In[218]:


def run_style_transfer(modle,style_losses,content_losses,input_img,num_steps=300,
                      style_weight = 1000000,content_weight=1):
    optimizer = get_input_optimizer(input_img)
    
    # Optimisation starts here 
    run = [0]
    
    while run[0] <= num_steps :
        
        def closure():
            
            input_img.data.clamp_(0,1)
            
            optimizer.zero_grad()
            model(input_img)
            
            style_score = 0 
            content_score = 0 
            
            for style_layer in style_losses:
                style_score += (1/5)*style_layer.loss
                
            for content_layer in content_losses:
                content_score += content_layer.loss
                
            style_score *= style_weight
            content_score *= content_weight
            
            loss = style_score + content_score
            loss.backward()
            
            run[0]+=1 
            
            if run[0] %50 ==0 :
                print("run {}:".format(run))
            
            
            
            return style_score + content_score 
        
        optimizer.step(closure)
        
        # making tensor btw 0 and 1
        input_img.data.clamp_(0,1)
        
        return input_img


# In[219]:


# Images 
# Resizing the image 
# imgsize = 512  if torch.cuda.is_available() else 128 
imgsize = 512 

content_img = image_loader("/Users/lakshaybhadana/Desktop/Virat.jpg",imgsize=512)
style_img = image_loader("/Users/lakshaybhadana/Desktop/Cartoon.jpg",imgsize=512)
print(content_img.size())
print(style_img.size())

# sanity check or validate a condition (We need to import both as of the same size)
assert content_img.size() == style_img.size()


model,style_losses,content_losses = get_style_model_losses(cnn,cnn_normalization_mean,cnn_normalization_std,style_img,content_img)

output = run_style_transfer(model,style_losses,content_losses,input_img,num_steps=300,
                      style_weight = 1000000,content_weight=1)

imshow(output)


# In[92]:


# image = Image.open("/Users/lakshaybhadana/Desktop/Cartoon.jpg")
# image


# In[ ]:




