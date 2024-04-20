import torch
from utils.metrics import id_correlation, distance_correlation, rbf_cka, linear_cka
import matplotlib.pyplot as plt
from tqdm import tqdm
from anatome.similarity import svcca_distance
import dcor
import torchvision

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(torch.flatten)])
trainset = torchvision.datasets.MNIST(f'./data/', transform=transform,  train=True, download=True)
testset = torchvision.datasets.MNIST(f'./data/', transform=transform, train=False, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)  
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)

class MLP(torch.nn.Module):
    def __init__(self, num_layers, slope):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.slope = slope
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(torch.nn.Linear(784, 784, bias=False))
        self.head = torch.nn.Linear(dim, 10)
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.nn.functional.leaky_relu(layer(x), self.slope)
        rep = x.cpu()
        x = self.head(x)
        return x, rep
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
idcorr = []
dcorr = []
cka = []
cca = []
for slope in tqdm(torch.arange(1, -0.1, -0.1)):
    model = MLP(10, slope)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.nn.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            out, _ = model(x)
            l = loss(out, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
    with torch.no_grad():
        x, y = next(iter(testloader))
        x=x.to(device)
        out, rep = model(x.to(device))
        print("Acc: ", torch.sum(torch.argmax(out, dim=1)==y.to(device)).item()/10000)
        idcorr.append(id_correlation(rep.to(device), x, N=100)['p'])
        dcorr.append(distance_correlation(rep.to(device), x))
        cka.append(linear_cka(rep.to(device), x).cpu())
        cca.append(1-svcca_distance(rep.to(device), x, accept_rate=0.99, backend='svd').cpu())
    
exit()
for x, y in trainloader:
        x = x.view(x.size(0), -1)
        embedding1 = model(x)
        idcorr.append(id_correlation(embedding1, x, N=100)['p'])
        dcorr.append(distance_correlation(embedding1, x))
        cka.append(linear_cka(embedding1, x).cpu())
        cca.append(1-svcca_distance(embedding1, x, accept_rate=0.99, backend='svd').cpu())
        break

with torch.no_grad():
    dim=512
    data=torch.randn(10000, dim)

    #write a simple neural network that reduces the dimensionality to 2
    class MLP(torch.nn.Module):
        def __init__(self, dim, num_layers):
            super(MLP, self).__init__()
            self.num_layers = num_layers

            self.layers = torch.nn.ModuleList()
            for _ in range(num_layers):
                self.layers.append(torch.nn.Linear(dim, dim, bias=False))
            
            # Define output layer
            
        def forward(self, x, slope):
            
            # Hidden layers
            for layer in self.layers[:-1]:
                x = torch.nn.functional.leaky_relu(layer(x), slope)
            x = self.layers[-1](x)
            return x

    idcorr=[]
    dcorr=[]
    cca=[]
    model = MLP(dim, 10)
    for slope in tqdm(torch.arange(1, -0.01, -1)):
        embedding1 = model.to('cuda')(data.to('cuda'), slope)
        #embedding2 = Net(128, slope)(data)
        idcorr.append(id_correlation(embedding1, data.to('cuda'), N=100)['p'])
        dcorr.append(distance_correlation(embedding1, data.to('cuda')))
        print(dcorr[-1])
        print(dcor.distance_correlation(embedding1.cpu(), data))
        cca.append(1-svcca_distance(embedding1, data.to('cuda'), accept_rate=0.99, backend='svd').cpu())
        #cka.append(linear_cka(embedding1, data.to('cuda')).cpu())
    plt.plot(idcorr, label='p-value (ours)')
    plt.plot(dcorr, label='distance correlation')
    plt.plot(cca, label='cca')
    plt.legend()