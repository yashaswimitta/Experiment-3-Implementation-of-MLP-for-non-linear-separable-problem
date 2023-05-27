# Experiment-3-Implementation-of-MLP-for-non-linear-separable-problem
**AIM:**

To implement a perceptron for classification using Python

**EQUIPMENTS REQUIRED:**
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

**RELATED THEORETICAL CONCEPT:**
Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows
XOR truth table
![Img1](https://user-images.githubusercontent.com/112920679/195774720-35c2ed9d-d484-4485-b608-d809931a28f5.gif)

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below

![Img2](https://user-images.githubusercontent.com/112920679/195774898-b0c5886b-3d58-4377-b52f-73148a3fe54d.gif)

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.To separate the two outputs using linear equation(s), it is required to draw two separate lines as shown in figure below:
![Img 3](https://user-images.githubusercontent.com/112920679/195775012-74683270-561b-4a3a-ac62-cf5ddfcf49ca.gif)
For a problem resembling the outputs of XOR, it was impossible for the machine to set up an equation for good outputs. This is what led to the birth of the concept of hidden layers which are extensively used in Artificial Neural Networks. The solution to the XOR problem lies in multidimensional analysis. We plug in numerous inputs in various layers of interpretation and processing, to generate the optimum outputs.
The inner layers for deeper processing of the inputs are known as hidden layers. The hidden layers are not dependent on any other layers. This architecture is known as Multilayer Perceptron (MLP).
![Img 4](https://user-images.githubusercontent.com/112920679/195775183-1f64fe3d-a60e-4998-b4f5-abce9534689d.gif)
The number of layers in MLP is not fixed and thus can have any number of hidden layers for processing. In the case of MLP, the weights are defined for each hidden layer, which transfers the signal to the next proceeding layer.Using the MLP approach lets us dive into more than two dimensions, which in turn lets us separate the outputs of XOR using multidimensional equations.Each hidden unit invokes an activation function, to range down their output values to 0 or The MLP approach also lies in the class of feed-forward Artificial Neural Network, and thus can only communicate in one direction. MLP solves the XOR problem efficiently by visualizing the data points in multi-dimensions and thus constructing an n-variable equation to fit in the output values using back propagation algorithm

**Algorithm :**

Step 1 : Initialize the input patterns for XOR Gate
Step 2: Initialize the desired output of the XOR Gate
Step 3: Initialize the weights for the 2 layer MLP with 2 Hidden neuron 
              and 1 output neuron
Step 3: Repeat the  iteration  until the losses become constant and 
              minimum
              (i)  Compute the output using forward pass output
              (ii) Compute the error  
		          (iii) Compute the change in weight ‘dw’ by using backward 
                     propagation algorithm.
             (iv) Modify the weight as per delta rule.
             (v)   Append the losses in a list
Step 4 : Test for the XOR patterns.

**PROGRAM ** 
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x=np.array([[0,0,1,1],[0,1,0,1]])
y=np.array([[0,1,1,0]])

#No. of neurons in each layer
n_x = 2 #Input-Layer
n_y = 1 #Output-Layer
n_h = 2 #Hidden-Layer

m = x.shape[1]
lr = 0.1 #Learning Rate
np.random.seed(2)
w1 = np.random.rand(n_h,n_x)   # Weight matrix for hidden layer (x to hidden)
w2 = np.random.rand(n_y,n_h)   # Weight matrix for output layer (hidden to y)
losses = []

def sigmoid(z):
    z= 1/(1+np.exp(-z))
    return z

def forward_prop(w1,w2,x):
    z1 = np.dot(w1,x)   #y1 - Summing junction for hidden layer (v1.w1+v2.w1)
    a1 = sigmoid(z1)    #Hidden layer activation function - phi(.)

    z2 = np.dot(w2,a1) #y3 - Summing junction for output layer (v3.w3)
    a2 = sigmoid(z2)   #Output layer activation function - phi(.)
    return z1,a1,z2,a2

def back_prop(m,w1,w2,z1,a1,z2,a2,y):    
    dz2 = a2-y  #Calculating error actual - desired

    dw2 = np.dot(dz2,a1.T)/m  #adjusting output layer weight matrix
    dz1 = np.dot(w2.T,dz2) * a1*(1-a1) #adjusting output layer

    dw1 = np.dot(dz1,x.T)/m  #adjusting hidden layer weight matrix

    dw1 = np.reshape(dw1,w1.shape)
    dw2 = np.reshape(dw2,w2.shape)    
    return dz2,dw2,dz1,dw1

iterations = 10000
for i in range(iterations):
    z1,a1,z2,a2 = forward_prop(w1,w2,x)

    loss = -(1/m)*np.sum(y*np.log(a2)+(1-y)*np.log(1-a2))
    losses.append(loss)
    da2,dw2,dz1,dw1 = back_prop(m,w1,w2,z1,a1,z2,a2,y)
    
    w2 = w2-lr*dw2
    w1 = w1-lr*dw1

# We plot losses to see how our network is doing
plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")

def predict(w1,w2,input):
    z1,a1,z2,a2 = forward_prop(w1,w2,test)
    a2 = np.squeeze(a2)
    if a2>=0.5:
        print( [i[0] for i in input], 1)
    else:
        print( [i[0] for i in input], 0)

print('Input',' Output')
test=np.array([[0],[0]])
predict(w1,w2,test)
test=np.array([[0],[1]])
predict(w1,w2,test)
test=np.array([[1],[0]])
predict(w1,w2,test)
test=np.array([[1],[1]])
predict(w1,w2,test)
```


 **OUTPUT** 
 ![nn 4 1](https://github.com/yashaswimitta/Experiment-3-Implementation-of-MLP-for-non-linear-separable-problem/assets/94619247/141ee168-6b15-4d72-8667-b7abd1bc71e2)


**RESULT**
Thus a MLP is implemented for non linear separable problem using Python.
