from MD_to_CVAE.MD_to_CVAE_scripts import cm_to_cvae as t_cm_to_cvae
from MD_to_CVAE.utils import cm_to_cvae as n_cm_to_cvae
import numpy as np
import torch

n_cm = np.round(np.random.rand(210,100))
t_cm = torch.from_numpy(n_cm)

n_res = n_cm_to_cvae([n_cm])

t_res = t_cm_to_cvae(t_cm)

print(np.squeeze(n_res))
print(torch.squeeze(t_res))

print(torch.norm(torch.from_numpy(n_res)-t_res))