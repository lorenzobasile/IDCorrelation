import torch
from utils import metrics
from tqdm import tqdm, trange
from anatome.similarity import svcca_distance
import torchvision
import os

torch.manual_seed(0)


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(torch.flatten)])
testset = torchvision.datasets.MNIST('./data/', transform=transform, train=False, download=True)
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
rand_inits = 10
slopes = torch.arange(1, -0.01, -0.02)
idcor = torch.zeros(rand_inits, len(slopes))
pvalues = torch.zeros(rand_inits, len(slopes))
dcor = torch.zeros(rand_inits, len(slopes))
rbf_cka = torch.zeros(rand_inits, len(slopes))
linear_cka = torch.zeros(rand_inits, len(slopes))
cca = torch.zeros(rand_inits, len(slopes))
for run in trange(rand_inits):
    model1 = MLP(15).to(device)
    for s,slope in enumerate(slopes):
        with torch.no_grad():
            x, y = next(iter(testloader))
            x=x.to(device)
            out, rep1 = model1(x.to(device), slope)
            corr=metrics.id_correlation(rep1.to(device), x.to(device), 20)
            idcor[run, s] = corr['corr']
            pvalues[run, s] = corr['p']
            dcor[run, s] = metrics.distance_correlation(rep1.to(device), x.to(device))
            rbf_cka[run,s]=metrics.rbf_cka(rep1.to(device), x.to(device)).cpu()
            linear_cka[run,s]=metrics.linear_cka(rep1.to(device), x.to(device)).cpu()
            cca[run,s]=1-svcca_distance(rep1.to(device), x.to(device), accept_rate=0.99, backend='svd').cpu()


torch.save(idcor, 'results/mnist/idcor.pt')
torch.save(pvalues, 'results/mnist/pvalues.pt')
torch.save(dcor, 'results/mnist/dcor.pt')
torch.save(rbf_cka, 'results/mnist/rbf_cka.pt')
torch.save(linear_cka, 'results/mnist/linear_cka.pt')
torch.save(cca, 'results/mnist/svcca.pt')

