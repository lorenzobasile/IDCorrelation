import torch
from utils import metrics
from tqdm import tqdm
from anatome.similarity import svcca_distance
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
        self.head = torch.nn.Linear(784, 10)
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.nn.functional.leaky_relu(layer(x), self.slope)
        rep = x.cpu()
        x = self.head(x)
        return x, rep
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
idcorr = []
pvalues = []
dcor = []
rbf_cka = []
linear_cka = []
cca = []
for slope in tqdm(torch.arange(1, -0.01, -0.01)):
    model = MLP(10, slope).to(device)
    loss = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        x, y = next(iter(testloader))
        x=x.to(device)
        out, rep = model(x.to(device))
        icorrdc=metrics.id_correlation(rep.to(device), x, 200)
        idcorr.append(corr['corr'])
        pvalues.append(corr['p'])
        dcor.append(metrics.distance_correlation(rep.to(device), x))
        rbf_cka.append(metrics.rbf_cka(rep.to(device), x).cpu())
        linear_cka.append(metrics.linear_cka(rep.to(device), x).cpu())
        cca.append(1-svcca_distance(rep.to(device), x, accept_rate=0.99, backend='svd').cpu())
idcorr=torch.tensor(idcorr)
pvalues=torch.tensor(pvalues)
dcor=torch.tensor(dcor)
rbf_cka=torch.tensor(rbf_cka)
linear_cka=torch.tensor(linear_cka)
cca=torch.tensor(cca)

torch.save(idcorr, 'results/mnist/idcorr.pt')
torch.save(pvalues, 'results/mnist/pvalues.pt')
torch.save(dcor, 'results/mnist/dcor.pt')
torch.save(rbf_cka, 'results/mnist/rbf_cka.pt')
torch.save(linear_cka, 'results/mnist/linear_cka.pt')
torch.save(cca, 'results/mnist/svcca.pt')


