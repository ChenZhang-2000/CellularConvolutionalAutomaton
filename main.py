import numpy as np
import torch

from model import evolve, Evolution
from data import Fungi

device = 'cuda:0'
data = Fungi()

# only uncomment this line in the first time to run the neural networks
# data.fitting(range(34))

bs = 34
simulation = Evolution(data=data, dim=34, batch_size=bs,
                       start_temp=16., start_humidity=-2.0,
                       max_temp=22., max_humidity=0.,
                       min_temp=10., min_humidity=-5.,
                       temp_rand_step=5, humid_rand_step=2,
                       board_size=128, device=device)
simulation.evolution(300)
