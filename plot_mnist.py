import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import StrMethodFormatter

plt.style.use('tableau-colorblind10')



idcor=torch.load('results/mnist/idcor.pt')
dcor=torch.load('results/mnist/dcor.pt')
rbf_cka=torch.load('results/mnist/rbf_cka.pt')
linear_cka=torch.load('results/mnist/linear_cka.pt')
svcca=torch.load('results/mnist/svcca.pt')
plt.figure(figsize=(13.1, 6))
x_values = torch.arange(0.00, 1.01, 0.02)[::2]

# Plot each curve with shaded error regions
plt.plot(x_values, idcor.mean(dim=0)[::2], 'o-', label='$I_d$Cor')
plt.fill_between(
    x_values,
    idcor.mean(dim=0)[::2] - idcor.std(dim=0)[::2],   # Lower bound
    idcor.mean(dim=0)[::2] + idcor.std(dim=0)[::2],   # Upper bound
    alpha=0.2  # Transparency for the shaded area
)

plt.plot(x_values, dcor.mean(dim=0)[::2], 'o-', label='dCor')
plt.fill_between(
    x_values,
    dcor.mean(dim=0)[::2] - dcor.std(dim=0)[::2],
    dcor.mean(dim=0)[::2] + dcor.std(dim=0)[::2],
    alpha=0.2
)

plt.plot(x_values, linear_cka.mean(dim=0)[::2], 'o-', label='CKA (linear)')
plt.fill_between(
    x_values,
    linear_cka.mean(dim=0)[::2] - linear_cka.std(dim=0)[::2],
    linear_cka.mean(dim=0)[::2] + linear_cka.std(dim=0)[::2],
    alpha=0.2
)

plt.plot(x_values, rbf_cka.mean(dim=0)[::2], 'o-', label='CKA (RBF)')
plt.fill_between(
    x_values,
    rbf_cka.mean(dim=0)[::2] - rbf_cka.std(dim=0)[::2],
    rbf_cka.mean(dim=0)[::2] + rbf_cka.std(dim=0)[::2],
    alpha=0.2
)

plt.plot(x_values, svcca.mean(dim=0)[::2], 'o-', label='SVCCA')
plt.fill_between(
    x_values,
    svcca.mean(dim=0)[::2] - svcca.std(dim=0)[::2],
    svcca.mean(dim=0)[::2] + svcca.std(dim=0)[::2],
    alpha=0.2
)
#plt.plot(torch.arange(0.00, 1.01, 0.02)[::2], idcor.mean(dim=0)[::2], 'o-', label='$I_d$Cor')
#plt.plot(torch.arange(0.00, 1.01, 0.02)[::2], dcor.mean(dim=0)[::2], 'o-', label='dCor')
#plt.plot(torch.arange(0.00, 1.01, 0.02)[::2], linear_cka.mean(dim=0)[::2], 'o-', label='CKA (linear)')
#plt.plot(torch.arange(0.00, 1.01, 0.02)[::2], rbf_cka.mean(dim=0)[::2], 'o-', label='CKA (RBF)')
#plt.plot(torch.arange(0.00, 1.01, 0.02)[::2], svcca.mean(dim=0)[::2], 'o-', label='SVCCA')
plt.legend(fontsize=13.5, loc='lower left')
plt.yticks(fontsize=12)
plt.xticks(ticks=np.arange(0.0, 1.1, 0.2), labels=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0], fontsize=12)
plt.ylabel('Correlation', fontsize=20)
plt.xlabel('LeakyReLU slope', fontsize=20)

plt.savefig('results/mnist.svg', dpi=200, bbox_inches='tight', format='svg')

plt.figure(figsize=(6, 6))
out=torch.arange(-1.0, 1.01, 0.01)
plt.plot(torch.arange(-1.0, 1.01, 0.01), out, '-', linewidth=6.0, c='#FF800E')
plt.ylim(-1, 1)
plt.xlim(-1, 1)
plt.axhline(0, color='black', linewidth=0.5)  # horizontal line at y=0
plt.axvline(0, color='black', linewidth=0.5)  # vertical line at x=0
plt.xticks([], [])
plt.yticks([], [])
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.savefig('results/identity.svg', dpi=200, bbox_inches='tight', format='svg')

plt.figure(figsize=(6, 6))
out=torch.arange(-1.0, 1.01, 0.01)
out[out<0]=0
plt.plot(torch.arange(-1.0, 1.01, 0.01), out, '-', linewidth=6.0, c='#FF800E')
plt.ylim(-1, 1)
plt.xlim(-1, 1)
plt.axhline(0, color='black', linewidth=0.5)  # horizontal line at y=0
plt.axvline(0, color='black', linewidth=0.5)  # vertical line at x=0
plt.xticks([], [])
plt.yticks([], [])
for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.savefig('results/relu.svg', dpi=200, bbox_inches='tight', format='svg')

plt.figure(figsize=(6, 6))
out=torch.arange(-1.0, 1.01, 0.01)
out[out<0]=out[out<0]*0.5
plt.plot(torch.arange(-1.0, 1.01, 0.01), out, '-', linewidth=6.0, c='#FF800E')
plt.ylim(-1, 1)
plt.xlim(-1, 1)
plt.axhline(0, color='black', linewidth=0.5)  # horizontal line at y=0
plt.axvline(0, color='black', linewidth=0.5)  # vertical line at x=0
plt.xticks([], [])
plt.yticks([], [])
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.savefig('results/half_relu.svg', dpi=200, bbox_inches='tight', format='svg')
