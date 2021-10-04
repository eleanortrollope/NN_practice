import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    # 1 input image channel, 6 output channels, 5x5 square convolution 
    # kernel 
    self.conv1 = nn.Conv2d(1,6,5)
    self.conv2 = nn.Conv2d(6,16,5)
    # an affine operation: y = Wx + b 
    self.fc1 = nn.Linear(16*5*5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    # max pooling over a (2, 2) window 
    x = F.max_pool2d(F.relu(self.conv1(x)),(2, 2))
    # if the size is a square, you can specify with a single number 
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = torch.flatten(x, 1) # flatten all dimensions except the batch dimensio 
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x 

net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10) # a dummy target, for example
target = target.view(1, -1) # make it the same shape as the output 
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn) # MSE loss
print(loss.grad_fn.next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU 


import torch.optim as optim 

# create your optimiser 

optimizer = optim.SGD(net.parameters(), lr = 0.01)

# in your training loop 

optimizer.zero_grad() # zeros the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # does the update 
