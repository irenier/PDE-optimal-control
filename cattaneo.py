import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from deepxde import geometry
import os
import itertools

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
torch.set_default_dtype(torch.float32)
dtype = np.float32
np.random.seed(1111)
torch.manual_seed(1111)

# colorbar 设置


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="3%")
    return fig.colorbar(mappable, cax=cax)


# 创建文件夹存储图片
fig_path = './figures/'
# 判断是否存在文件夹如果不存在则创建为文件夹
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

# 参数设置
tau = 1.0
omega = 1.0
time_T = 1.0


def Y_d_fun(X):
    t = X[:, 2:3]
    delta_1 = 0.5 + 0.25 * torch.cos(2. * torch.pi * t)
    delta_2 = 0.5 + 0.25 * torch.sin(2. * torch.pi * t)
    return torch.exp(-20.0 * ((X[:, 0:1] - delta_1) ** 2 + (X[:, 1:2] - delta_2) ** 2))


class Cattaneo:
    def __init__(self):

        self.learning_rate_Y = 1e-3
        self.learning_rate_P = 1e-3

        self.Y_dim = 3
        self.P_dim = 3

        self.iter_warm = 500
        self.iter_epoch = 1500
        self.iter_finetune = 50
        self.display_every = 20

        self.L_train_warm_n = {
            'interior': 50 ** 3,
            'boundary': 4 * 50 * 50,
            'initial': 50 ** 2,
        }

        self.L_train_n = {
            'interior': 10 ** 3,
            'boundary': 4 * 10 * 10,
            'initial': 10 ** 2,
        }

        self.L_array = list()

        self.activation = nn.Tanh
        self.neurons = 64
        self.scheduler_activation = False

        self.model_Y = nn.Sequential(
            nn.Linear(self.Y_dim, self.neurons, device=device),
            self.activation(),
            nn.Linear(self.neurons, self.neurons, device=device),
            self.activation(),
            nn.Linear(self.neurons, self.neurons, device=device),
            self.activation(),
            nn.Linear(self.neurons, self.neurons, device=device),
            self.activation(),
            nn.Linear(self.neurons, 1, device=device)
        )
        self.model_P = nn.Sequential(
            nn.Linear(self.P_dim, self.neurons, device=device),
            self.activation(),
            nn.Linear(self.neurons, self.neurons, device=device),
            self.activation(),
            nn.Linear(self.neurons, self.neurons, device=device),
            self.activation(),
            nn.Linear(self.neurons, self.neurons, device=device),
            self.activation(),
            nn.Linear(self.neurons, 1, device=device)
        )

        def initializer(model):
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight)
                nn.init.zeros_(model.bias)

        self.model_Y.apply(initializer)
        self.model_P.apply(initializer)

        # 定义优化器
        self.optimizer_Y = optim.Adam(
            self.model_Y.parameters(), lr=self.learning_rate_Y)
        self.optimizer_P = optim.Adam(
            self.model_P.parameters(), lr=self.learning_rate_P)
        self.criterion = nn.MSELoss()

        # 制定 scheduler
        self.scheduler_Y = optim.lr_scheduler.StepLR(
            self.optimizer_Y, step_size=200, gamma=0.5)
        self.scheduler_P = optim.lr_scheduler.StepLR(
            self.optimizer_P, step_size=200, gamma=0.5)

        self.geomtime = geometry.GeometryXTime(
            geometry.Rectangle((0, 0), (1, 1)), geometry.TimeDomain(0, time_T))

    def model_U(self, X):
        return -self.model_P(X) / omega

    def loss_Y(self, X_boundary, X_initial, X_interior, verbose=False):

        Y = self.model_Y(X_interior)
        U = self.model_U(X_interior)

        # 计算 pde 损失：loss_F
        dYdX = torch.autograd.grad(
            Y, X_interior,
            grad_outputs=torch.ones_like(Y, device=device),
            retain_graph=True,
            create_graph=True
        )[0]

        dYdx = dYdX[:, 0:1]
        dYdy = dYdX[:, 1:2]
        dYdt = dYdX[:, 2:3]

        dYdxdX = torch.autograd.grad(
            dYdx, X_interior,
            grad_outputs=torch.ones_like(dYdx, device=device),
            retain_graph=True,
            create_graph=True
        )[0]
        dYdydX = torch.autograd.grad(
            dYdy, X_interior,
            grad_outputs=torch.ones_like(dYdy, device=device),
            retain_graph=True,
            create_graph=True
        )[0]
        dYdtdX = torch.autograd.grad(
            dYdt, X_interior,
            grad_outputs=torch.ones_like(dYdt, device=device),
            retain_graph=True,
            create_graph=True
        )[0]
        dYdxdx = dYdxdX[:, 0:1]
        dYdydy = dYdydX[:, 1:2]
        dYdtdt = dYdtdX[:, 2:3]

        loss_F = self.criterion(tau * dYdtdt + dYdt - dYdxdx - dYdydy, U)

        # 计算边界损失：loss_B
        Y_boundary = self.model_Y(X_boundary)
        Y_initial = self.model_Y(X_initial)

        dYdX_initial = torch.autograd.grad(
            Y_initial, X_initial,
            grad_outputs=torch.ones_like(Y_initial, device=device),
            retain_graph=True,
            create_graph=True
        )[0]
        dYdt_initial = dYdX_initial[:, 2:3]

        loss_B = torch.mean(
            Y_boundary ** 2) + torch.mean(Y_initial ** 2) + torch.mean(dYdt_initial ** 2)

        loss = loss_F + loss_B

        if verbose:
            print("Y. loss_F: %.5e, loss_B: %.5e, loss: %.5e" %
                  (loss_F, loss_B, loss), end="   ")
        return loss

    def loss_P(self, X_boundary, X_final, X_interior, verbose=False):

        P = self.model_P(X_interior)
        Y = self.model_Y(X_interior)
        Y_d = Y_d_fun(X_interior)

        # 计算 pde 损失：loss_F
        dPdX = torch.autograd.grad(
            P, X_interior,
            grad_outputs=torch.ones_like(P, device=device),
            retain_graph=True,
            create_graph=True
        )[0]

        dPdx = dPdX[:, 0:1]
        dPdy = dPdX[:, 1:2]
        dPdt = dPdX[:, 2:3]

        dPdxdX = torch.autograd.grad(
            dPdx, X_interior,
            grad_outputs=torch.ones_like(dPdx, device=device),
            retain_graph=True,
            create_graph=True
        )[0]
        dPdydX = torch.autograd.grad(
            dPdy, X_interior,
            grad_outputs=torch.ones_like(dPdy, device=device),
            retain_graph=True,
            create_graph=True
        )[0]
        dPdtdX = torch.autograd.grad(
            dPdt, X_interior,
            grad_outputs=torch.ones_like(dPdt, device=device),
            retain_graph=True,
            create_graph=True
        )[0]
        dPdxdx = dPdxdX[:, 0:1]
        dPdydy = dPdydX[:, 1:2]
        dPdtdt = dPdtdX[:, 2:3]

        loss_F = self.criterion(tau * dPdtdt - dPdt - dPdxdx - dPdydy, Y - Y_d)

        # 计算边界损失：loss_B
        P_boundary = self.model_P(X_boundary)
        P_final = self.model_P(X_final)

        dPdX_final = torch.autograd.grad(
            P_final, X_final,
            grad_outputs=torch.ones_like(P_final, device=device),
            retain_graph=True,
            create_graph=True
        )[0]
        dPdt_final = dPdX_final[:, 2:3]

        loss_B = torch.mean(
            P_boundary ** 2) + torch.mean(P_final ** 2) + torch.mean(dPdt_final ** 2)

        loss = loss_F + loss_B

        if verbose:
            print("P. loss_F: %.5e, loss_B: %.5e, loss: %.5e" %
                  (loss_F, loss_B, loss), end="   ")
        return loss

    def train_YP(self, X_boundary, X_initial, X_final, X_interior, verbose=False):
        self.optimizer_Y.zero_grad()
        self.optimizer_P.zero_grad()
        L_Y = self.loss_Y(X_boundary, X_initial, X_interior, verbose)
        L_P = self.loss_P(X_boundary, X_final, X_interior, verbose)
        L = L_Y + L_P

        # save loss
        if verbose:
            self.L_array.append(L.item())

        L.backward()
        self.optimizer_Y.step()
        self.optimizer_P.step()

        # 是否使用 scheduler
        if self.scheduler_activation:
            self.scheduler_Y.step()
            self.scheduler_P.step()

    def train(self, format='png', verbose=False):
        # 预训练
        X_boundary, X_initial, X_final, X_interior = self.generate_uniform_data(
            self.L_train_warm_n)
        for i_warm in range(self.iter_warm):
            # 生成数据
            if (i_warm + 1) % self.display_every == 0:
                print("iter %3d.  " % (i_warm + 1), end="")
                self.train_YP(X_boundary, X_initial, X_final, X_interior, True)
                print("\n", end="")
            else:
                self.train_YP(X_boundary, X_initial,
                              X_final, X_interior, False)

        self.draw('y_d', format=format)
        if verbose:
            for var in ['y', 'u', 'p']:
                self.draw(var, format=format)

        for i_epoch in range(self.iter_epoch):
            print("\nepoch %3d" % (i_epoch + 1))
            # 生成数据
            X_boundary, X_initial, X_final, X_interior = self.generate_random_data(
                self.L_train_n)

            # 利用 verbose 进行观察
            if verbose:
                for i_finetune in range(self.iter_finetune - 1):
                    if i_finetune % self.display_every == 0:
                        print("iter %3d.  " % (i_finetune + 1), end="")
                        self.train_YP(X_boundary, X_initial,
                                      X_final, X_interior, True)
                        print("\n", end="")
                    else:
                        self.train_YP(X_boundary, X_initial,
                                      X_final, X_interior, False)
                print("Final.     ", end="")
                self.train_YP(X_boundary, X_initial, X_final, X_interior, True)
                print("\n", end="")
            else:
                for i_finetune in range(self.iter_finetune - 1):
                    self.train_YP(X_boundary, X_initial,
                                  X_final, X_interior, False)
                print("Final.     ", end="")
                self.train_YP(X_boundary, X_initial, X_final, X_interior, True)
                print("\n", end="")

        for var in ['y', 'u', 'p']:
            self.draw(var, format=format)

    def plot_data(self, func, t):
        # 设置画图相关参数
        N = 200
        temp_x = np.linspace(0., 1., N, dtype=dtype)
        temp_y = np.linspace(0., 1., N, dtype=dtype)
        x, y = np.meshgrid(temp_x, temp_y)

        temp_xy = np.dstack((x, y)).reshape(-1, 2)

        X = []
        for item in t:
            temp_X = np.insert(temp_xy, 2, item, axis=1)
            X.append(torch.tensor(temp_X, device=device))

        with torch.no_grad():
            Y = []
            for item in X:
                Y.append(func(item).reshape(N, N).cpu().numpy())

        return x, y, Y

    def draw(self, var, t=np.linspace(time_T / 4, time_T, 4), format='png', save=True):

        if var == 'y':
            func = self.model_Y
            name = 'state'
        elif var == 'u':
            func = self.model_U
            name = 'control'
        elif var == 'p':
            func = self.model_P
            name = 'adjoint_state'
        elif var == 'y_d':
            func = Y_d_fun
            name = 'y_d'
        else:
            print("Can't plot variable %s" % name)

        x, y, Y = self.plot_data(func, t)

        N_t = len(t)
        cols = 2
        fig, ax = plt.subplots(int(N_t / cols), cols)
        fig.subplots_adjust(wspace=0.4, hspace=0.6)

        # 画 t 时刻的 state
        for i in range(N_t):
            r, c = int(i / cols), i % cols
            ax[r, c].set_aspect('equal')
            cs = ax[r, c].pcolormesh(x, y, Y[i], cmap=cm.coolwarm)
            ax[r, c].set(xlabel='x', ylabel='y',
                         title=f'{name} when t = {t[i]}')
            colorbar(cs)

        plt.show()

        if save:
            s = './figures/' + name + time.strftime('-%d-%H-%M-%S',
                                                    time.localtime()) + '.' + format
            plt.savefig(s)

        plt.close()

    def save_model(self, name):
        # 创建文件夹存储图片
        model_path = './model/' + name + '/'
        # 判断是否存在文件夹如果不存在则创建为文件夹
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.save(self.model_P.state_dict(), model_path + 'model_P.pt')
        torch.save(self.model_Y.state_dict(), model_path + 'model_Y.pt')

    def load_model(self, name):
        model_path = './model/' + name + '/'

        model_P_state_dict = torch.load(model_path + 'model_P.pt')
        model_Y_state_dict = torch.load(model_path + 'model_Y.pt')

        self.model_P.load_state_dict(model_P_state_dict)
        self.model_Y.load_state_dict(model_Y_state_dict)

    def generate_uniform_data(self, train_n: dict):
        X_boundary = self.geomtime.uniform_boundary_points(
            train_n['boundary']).astype(dtype=dtype)

        X_x = self.geomtime.geometry.uniform_points(
            train_n['initial']).astype(dtype=dtype)

        t_initial = self.geomtime.timedomain.t0
        t_final = self.geomtime.timedomain.t1

        X_initial = np.hstack(
            (X_x, np.full([train_n['initial'], 1], t_initial, dtype=dtype)))
        X_final = np.hstack(
            (X_x, np.full([train_n['initial'], 1], t_final, dtype=dtype)))

        X_interior = self.geomtime.uniform_points(
            train_n['interior'], boundary=False).astype(dtype=dtype)
        return torch.tensor(X_boundary, device=device, requires_grad=True), torch.tensor(X_initial, device=device, requires_grad=True), torch.tensor(X_final, device=device, requires_grad=True), torch.tensor(X_interior, device=device, requires_grad=True)

    def generate_random_data(self, train_n: dict):
        X_boundary = self.geomtime.random_boundary_points(
            train_n['boundary']).astype(dtype=dtype)

        X_x = self.geomtime.geometry.random_points(
            train_n['initial']).astype(dtype=dtype)

        t_initial = self.geomtime.timedomain.t0
        t_final = self.geomtime.timedomain.t1

        X_initial = np.hstack(
            (X_x, np.full([train_n['initial'], 1], t_initial, dtype=dtype)))
        X_final = np.hstack(
            (X_x, np.full([train_n['initial'], 1], t_final, dtype=dtype)))

        X_interior = self.geomtime.random_points(
            train_n['interior']).astype(dtype=dtype)
        return torch.tensor(X_boundary, device=device, requires_grad=True), torch.tensor(X_initial, device=device, requires_grad=True), torch.tensor(X_final, device=device, requires_grad=True), torch.tensor(X_interior, device=device, requires_grad=True)

    def save_loss(self):

        N = np.arange(len(self.L_array)).reshape(-1, 1)
        loss = np.array(self.L_array).reshape(-1, 1)
        data = np.hstack((N, loss))
        np.savetxt("cattaneo-loss.dat", data)


bipinn = Cattaneo()
bipinn.train(verbose=False, format='pdf')
bipinn.save_model('cattaneo')
bipinn.save_loss()
