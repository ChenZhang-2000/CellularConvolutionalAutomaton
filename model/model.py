import torch
import numpy as np
from torch.nn.functional import relu, one_hot, tanh
from functools import reduce
from numba import jit
import PIL
from torchvision import transforms


small_corner_kernel = torch.tensor([[[0.54, 0.97, 0.54], [0.97, 1., 0.97], [0.54, 0.97, 0.54]]], device='cuda:0')
evolution_rate_kernel = torch.ones(3, 3, device='cuda:0') * 0.9
# less_center_kernel = torch.tensor([[[1.,1,1], [1,0.8,1], [1,1,1]]], device='cuda:0')


def normal(board, velocity):
    # x = board.ge(0.00).int() * board

    return -relu(-relu(board)+1)+1


def grow_kernel(r):
    global small_corner_kernel, evolution_rate_kernel
    v = (r/4).reshape(r.shape[0], 1, 1).repeat(1, 3, 3)  # *evolution_rate_kernel
    v[:, 1, 1] = 1
    return v * small_corner_kernel


def competition_kernel(r, winner_move, loser_move):
    # a = torch.ones(loser_move.shape[0], 3, 3).cuda()
    # print(a.shape, loser_move.shape)
    # a[:, 1, 1] = loser_move.reshape(loser_move.shape[0])
    return grow_kernel(r.reshape(r.shape[0], 1, 1) *
                       torch.logical_and(winner_move.eq(0), loser_move.eq(0)).int().cuda() + winner_move) + loser_move.repeat(1,3,3)


# @torch.jit.script
def evolve(board, velocity, ranking, device='cpu'):
    """
    :param board: growing tensor with shape (layers, height, width)
    :param velocity: velocity tensor with shape (layers,)
    :param device: device using
    :return:
    """
    precisions = 0.1
    dim = board.shape[0]
    covers = torch.sum(board, dim=0)
    area = torch.sum(torch.sum(board, dim=1), dim=1).reshape(dim, 1, 1).cuda()
    touched = torch.sum(covers.gt(board+precisions).int(), dim=0).ge(board.shape[0])
    padded_board = torch.zeros(dim, board.shape[1]+2, board.shape[2]+2, device='cuda:0')
    padded_board[:, 1:-1, 1:-1] = board
    new_board = torch.zeros(board.shape)
    forward_kernel = grow_kernel(velocity)
    # print(forward_kernel)
    for i in range(board.shape[1]):
        for j in range(board.shape[2]):
            vision = padded_board[:, i:i+3, j:j+3]
            edge = torch.sum(torch.sum(vision.ge(0.9).int(), dim=1), dim=1).gt(0).int()
            if touched[i, j]:
                competitors = board[:, i, j].gt(0).int().reshape(dim, 1, 1).cuda()
                competitor_area = competitors * area + competitors.eq(0).int().cuda()
                # print(torch.sum(torch.log(competitor_area)))
                # print(torch.log(competitor_area)) ranking
                probs = (competitor_area/torch.sum(competitor_area).cuda() + ranking/torch.sum(ranking).cuda())/2

                distribution = torch.distributions.categorical.Categorical(probs.reshape(dim,))
                winner = distribution.sample()
                move_v = (probs[winner] * velocity[winner]).reshape(())

                winner_onehot = one_hot(winner, num_classes=velocity.shape[0]).reshape(dim, 1, 1).cuda()
                winner_move = winner_onehot * move_v
                loser_move = (winner_onehot - competitors)*move_v
                kernel = competition_kernel(velocity, winner_move, loser_move, )
                new_board[:, i, j] = torch.sum(torch.sum(vision * kernel, dim=1),
                                               dim=1) * edge
            else:
                new_board[:, i, j] = torch.sum(torch.sum(vision * forward_kernel, dim=1),
                                               dim=1) * edge
    return normal(new_board, velocity)


class Evolution:

    def __init__(self, data, dim, batch_size, start_temp, start_humidity, max_temp, max_humidity,
                 min_temp, min_humidity, temp_rand_step, humid_rand_step, board_size, device='cpu'):
        self.data = data
        self.dim = dim
        self.bs = batch_size
        self.er = torch.ones(dim, 1, 1)

        self._temp = torch.tensor([start_temp, max_temp, min_temp])
        self.temp = torch.tensor([start_temp], device=device)
        self._humid = torch.tensor([start_humidity, max_humidity, min_humidity])
        self.humid = torch.tensor([start_humidity], device=device)
        self.er = 0

        self.temp_step = temp_rand_step
        self.humid_step = humid_rand_step
        self.device = device
        self.cuda = False if self.device == 'cpu' else True

        self.pop = torch.zeros((self.dim, 1), device=self.device)

        self.colors = self.data.colors.cuda()
        self.board_size = board_size
        self.board = torch.zeros(self.bs, self.board_size, self.board_size, device=self.device)

        self.weather_status = []
        self.area_status = []

    def evolution(self, rounds):
        er_temp_alpha, er_temp_beta, max_er_moisture, models, colors, moisture_niche, ranking = self.data.sampling(self.bs)
        # print(moisture_niche.shape)
        alpha = er_temp_alpha.cuda()
        beta = er_temp_beta.cuda()

        self.colors = colors.cuda()

        raw_er = tanh(beta * torch.exp(alpha * self.temp)).reshape(self.bs, 1)
        self.er = torch.cat([models[i](self.humid.reshape(1, 1).cuda()) for i in range(raw_er.shape[0])], dim=0) / \
                  max_er_moisture.cuda() * raw_er
        # print(self.er)
        self.board = torch.zeros(self.bs, self.board_size, self.board_size, device=self.device)
        for i in range(self.bs):
            x = np.random.randint(1, self.board_size-1)
            y = np.random.randint(0, self.board_size-1)
            self.board[i, x, y+1] = 0.97
            self.board[i, x, y] = 1.
            self.board[i, x, y-1] = 0.97
            self.board[i, x+1, y+1] = 0.54
            self.board[i, x+1, y] = 0.97
            self.board[i, x+1, y-1] = 0.54
            self.board[i, x-1, y+1] = 0.54
            self.board[i, x-1, y] = 0.97
            self.board[i, x-1, y-1] = 0.54
        # print(self.er.device)
        self.board = evolve(self.board, self.er.cuda(), ranking.reshape(self.bs, 1, 1), device=self.device)
        self.save_status(0)

        for i in range(rounds):

            delta_temp = self.temp_step * (torch.rand(1, device=self.device)*2-1)
            self.temp = self.temp + delta_temp.reshape(())
            if self.temp > self._temp[1]:
                self.temp = self._temp[1]
            elif self.temp < self._temp[2]:
                self.temp = self._temp[2]

            delta_humid = self.humid_step * (torch.rand(1, device=self.device)*2-1)
            self.humid = self.humid + delta_humid.reshape(())
            if self.humid > self._humid[1]:
                self.humid = self._humid[1]
            elif self.humid < self._humid[2]:
                self.humid = self._humid[2]

            raw_er = tanh(beta * torch.exp(alpha * self.temp)).reshape(self.bs, 1)
            self.er = torch.cat([models[i](self.humid.reshape(1, 1).cuda()) for i in range(raw_er.shape[0])], dim=0) / \
                      max_er_moisture.cuda() * raw_er
            # print(self.er)
            self.board = evolve(self.board, self.er, ranking.reshape(self.bs, 1, 1), device=self.device)
            self.save_status(i+1)
            torch.cuda.empty_cache()

    def save_status(self, round_num):
        self.weather_status.append([self.temp.detach(), self.humid.detach()])
        self.area_status.append(torch.sum(torch.sum(self.board, dim=1), dim=1))
        board = torch.sum(self.board.reshape(self.bs, 1, self.board_size, self.board_size).cuda() * self.colors, dim=0)

        transforms.ToPILImage()(board.cpu()).save(f'.\\temp\\img\\{round_num}.png')
