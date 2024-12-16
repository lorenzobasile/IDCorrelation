import torch
from utils import metrics
from tqdm import tqdm
from anatome.similarity import svcca_distance
import torchvision
import os

torch.manual_seed(0)


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(torch.flatten)])
testset = torchvision.datasets.MNIST(f'./data/', transform=transform, train=False, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)

if not os.path.exists('results/mnist'):
    os.makedirs('results/mnist')


class MLP(torch.nn.Module):
    def __init__(self, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(torch.nn.Linear(784, 784, bias=False))
        self.head = torch.nn.Linear(784, 10)
        
    def forward(self, x, slope):
        for layer in self.layers:
            x = torch.nn.functional.leaky_relu(layer(x), slope)
        rep = x.cpu()
        x = self.head(x)
        return x, rep
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
idcor = []
pvalues = []
dcor = []
rbf_cka = []
linear_cka = []
cca = []
model1 = MLP(15).to(device)
for slope in tqdm(torch.arange(1, -0.01, -0.02)):
    with torch.no_grad():
        x, y = next(iter(testloader))
        x=x.to(device)
        out, rep1 = model1(x.to(device), slope)
        corr=metrics.id_correlation(rep1.to(device), x.to(device), 2)
        idcor.append(corr['corr'])
        pvalues.append(corr['p'])
        print(corr)
        dcor.append(metrics.distance_correlation(rep1.to(device), x.to(device)))
        rbf_cka.append(metrics.rbf_cka(rep1.to(device), x.to(device)).cpu())
        linear_cka.append(metrics.linear_cka(rep1.to(device), x.to(device)).cpu())
        cca.append(1-svcca_distance(rep1.to(device), x.to(device), accept_rate=0.99, backend='svd').cpu())
        print(dcor[-1], cca[-1], rbf_cka[-1], linear_cka[-1])
idcor=torch.tensor(idcor)
pvalues=torch.tensor(pvalues)
dcor=torch.tensor(dcor)
rbf_cka=torch.tensor(rbf_cka)
linear_cka=torch.tensor(linear_cka)
cca=torch.tensor(cca)

torch.save(idcor, 'results/mnist/idcor.pt')
torch.save(pvalues, 'results/mnist/pvalues.pt')
torch.save(dcor, 'results/mnist/dcor.pt')
torch.save(rbf_cka, 'results/mnist/rbf_cka.pt')
torch.save(linear_cka, 'results/mnist/linear_cka.pt')
torch.save(cca, 'results/mnist/svcca.pt')


