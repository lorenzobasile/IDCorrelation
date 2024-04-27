import torch
import matplotlib.pyplot as plt
import seaborn as sns

idcor=torch.load('results/mnist/idcorr.pt')
dcor=torch.load('results/mnist/dcor.pt')
rbf_cka=torch.load('results/mnist/rbf_cka.pt')
linear_cka=torch.load('results/mnist/linear_cka.pt')
svcca=torch.load('results/mnist/svcca.pt')

plt.figure(figsize=(13.1, 6))
plt.plot(torch.arange(0.00, 1.01, 0.01)[::2], idcor[::2], label='IdCorr')
plt.plot(torch.arange(0.00, 1.01, 0.01)[::2], dcor[::2], label='dCor')
plt.plot(torch.arange(0.00, 1.01, 0.01)[::2], linear_cka[::2], label='CKA (linear)')
plt.plot(torch.arange(0.00, 1.01, 0.01)[::2], rbf_cka[::2], label='CKA (rbf)')
plt.plot(torch.arange(0.00, 1.01, 0.01)[::2], svcca[::2], label='SVCCA')
plt.legend()
plt.savefig('results/mnist.svg', dpi=200, bbox_inches='tight', format='svg')