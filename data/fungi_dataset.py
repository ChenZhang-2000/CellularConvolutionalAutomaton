import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn.functional import mse_loss
from torch.optim import Adam, SGD

epochs = 100
_traits_map_df = pd.read_csv('.\\data\\traits_map.csv')
traits_map = dict(zip(_traits_map_df.iloc[:, 0].values, _traits_map_df.iloc[:, 1].values))
# print(traits_map)

class FungusMoisture(nn.Module):
    def __init__(self):
        super(FungusMoisture, self).__init__()
        layers = [nn.Linear(1, 32), nn.ELU(), nn.Linear(32, 256), nn.ELU(), nn.Linear(256, 1024), nn.ELU(),
                  nn.Linear(1024, 1024), nn.ELU(), nn.Linear(1024, 256), nn.ELU(), nn.Linear(256, 32), nn.ELU(),
                  nn.Linear(32, 1), nn.ELU()]
        self.learner = nn.Sequential(*layers)

    def forward(self, x):
        return self.learner(x)

    def save(self, name):
        torch.save(self.state_dict(), f'.\\data\\er_wp_map\\{name}.txt')

    def load(self, name):
        self.load_state_dict(torch.load(f'.\\data\\er_wp_map\\{name}.txt'))
        self.eval()


class DataSet(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y)
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x.iloc[idx], self.y.iloc[idx]

    def __len__(self):
        return len(self.x)


class Fungi:
    def __init__(self):
        global traits_map

        er_temp = pd.read_csv('.\\data\\er_temp.csv')
        er_temp_slope = torch.tensor(er_temp.loc[:, 'slope'].values)
        er_temp_inter = torch.tensor(er_temp.loc[:, 'intercept'].values)
        # ranking =

        self.er_temp_alpha = er_temp_slope / torch.log(torch.tensor(10.))
        self.er_temp_beta = torch.exp(er_temp_inter / torch.log(torch.tensor(10.)))

        self.colors = torch.rand(34, 3, 1, 1, device='cuda:0')
        self.er_moisture_df = pd.read_csv('.\\data\\raw\\fungal_biogeography\\fungi_data\\Fungi_moisture_curves.csv')

        trait_df = pd.read_csv('.\\data\\raw\\Fungal_trait_data.csv')
        self.ranking = torch.tensor(trait_df.loc[:, 'ranking'].values, device='cuda:0')

        groups = self.er_moisture_df.groupby('species').groups
        self.max_er_moisture = torch.cat([torch.max(torch.tensor(self.er_moisture_df.iloc[groups[traits_map[i]], 2].values)).reshape(1,) for i in range(34)], dim=0).reshape(34,1)

        self.niche = torch.tensor(trait_df.loc[:, 'water.niche.width'].values)

        # comment this part out in the first time running
        # groups = self.er_moisture_df.groupby('species').groups
        # for i in range(34):
        #     survived = torch.tensor(self.er_moisture_df.iloc[groups[traits_map[i]], 2].values,
        #                             device='cuda:0').ge(self.max_er_moisture[i].cuda()/2)
        #     niche = torch.masked_select(torch.tensor(self.er_moisture_df.iloc[groups[traits_map[i]], 1].values, device='cuda:0'), survived)
        #     self.niche.append(torch.tensor([torch.max(niche), torch.min(niche)]).reshape(2, 1))
        #
        # self.niche = torch.cat(self.niche, dim=0)

    def fitting(self, idxes):
        global traits_map
        groups = self.er_moisture_df.groupby('species').groups
        # print(groups)
        for idx in idxes:
            i = groups[traits_map[idx]]
            specie = self.er_moisture_df.iloc[i, 1:]
            X, Y = specie.iloc[:, 0], specie.iloc[:, 1]

            data = DataSet(X, Y)
            train, test = random_split(data, [450, 58])
            network = FungusMoisture()
            network.cuda()
            optim = Adam(network.parameters(), lr=0.001)
            for j in range(epochs):
                for k, (x, y) in enumerate(DataLoader(train, batch_size=10)):
                    # print(k)
                    # print(x.reshape(10,1), y.reshape(10,1))
                    predict = network(x.reshape(x.shape[0], 1).float().cuda())
                    # print(predict)
                    optim.zero_grad()
                    loss = torch.sum((predict - y.reshape(x.shape[0], 1).cuda()) ** 2)
                    # print(loss)
                    loss.backward()
                    optim.step()
            network.save(idx)

    def read_fitted(self, idxes):
        models = []
        for idx in idxes:
            network = FungusMoisture()
            network.load(idx)
            network.cuda()
            models.append(network)
        return models

    def sampling(self, num):
        idxes = np.random.choice(np.arange(34), num)
        # print(idxes)
        return self.er_temp_alpha[idxes], self.er_temp_beta[idxes], self.max_er_moisture[idxes],  \
               self.read_fitted(idxes), self.colors[idxes], self.niche[idxes], self.ranking[idxes]
