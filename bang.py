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

# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
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
fig_path = "./figures/"
# 判断是否存在文件夹如果不存在则创建为文件夹
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

# 问题参数设置
beta = 1.0
time_T = 1.58


def Y_d_fun(X):
    return 0.5 * (1.0 - torch.pow(X[:, 0:1], 2))


class Bangbang:
    def __init__(self):
        super().__init__()

        self.learning_rate_Y = 1e-3
        self.learning_rate_P = 1e-2

        self.Y_dim = 2
        self.P_dim = 2

        self.iter_warm = 500
        self.iter_epoch = 500
        self.iter_finetune = 50
        self.display_every = 20

        self.L_train_warm_n = {
            "interior": 200 * 200,
            "boundary": 200 * 2,
            "initial": 200,
        }

        self.L_train_n = {"interior": 100 * 100, "boundary": 100 * 2, "initial": 100}

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
            nn.Linear(self.neurons, 1, device=device),
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
            nn.Linear(self.neurons, 1, device=device),
        )

        def initializer(model):
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight)
                nn.init.zeros_(model.bias)

        self.model_Y.apply(initializer)
        self.model_P.apply(initializer)

        # 定义优化器
        self.optimizer_Y = optim.Adam(
            self.model_Y.parameters(), lr=self.learning_rate_Y
        )
        self.optimizer_P = optim.Adam(
            self.model_P.parameters(), lr=self.learning_rate_P
        )
        self.criterion = nn.MSELoss()

        # 制定 scheduler
        self.scheduler_Y = optim.lr_scheduler.StepLR(
            self.optimizer_Y, step_size=200, gamma=0.5
        )
        self.scheduler_P = optim.lr_scheduler.StepLR(
            self.optimizer_P, step_size=200, gamma=0.5
        )

        self.geomtime = geometry.GeometryXTime(
            geometry.geometry_1d.Interval(0, 1), geometry.TimeDomain(0, time_T)
        )

    def model_U(self, X):
        P = self.model_P(X)
        U = -torch.tanh(1e8 * P)
        return U

    def loss_Y(self, X_boundary, X_initial, X_interior, verbose=False):
        # 计算右边界点的个数
        N_r = torch.sum(X_boundary[:, 0]).detach().cpu().numpy().astype(np.int32)

        Y = self.model_Y(X_interior)
        U = self.model_U(X_boundary[-N_r:, :])

        Y_boundary = self.model_Y(X_boundary)
        Y_initial = self.model_Y(X_initial)

        dYdX = torch.autograd.grad(
            Y,
            X_interior,
            grad_outputs=torch.ones_like(Y, device=device),
            retain_graph=True,
            create_graph=True,
        )[0]

        dYdx = dYdX[:, 0:1]
        dYdt = dYdX[:, 1:2]

        dYdxdX = torch.autograd.grad(
            dYdx,
            X_interior,
            grad_outputs=torch.ones_like(dYdx, device=device),
            retain_graph=True,
            create_graph=True,
        )[0]

        dYdxdx = dYdxdX[:, 0:1]

        # 处理边界条件
        dYdX_boundary = torch.autograd.grad(
            Y_boundary,
            X_boundary,
            grad_outputs=torch.ones_like(Y_boundary, device=device),
            retain_graph=True,
            create_graph=True,
        )[0]

        dYdx_boundary = dYdX_boundary[:, 0:1]

        loss_F = self.criterion(dYdt, dYdxdx)

        loss_B_L = torch.mean(dYdx_boundary[:-N_r] ** 2)
        loss_B_R = self.criterion(Y_boundary[-N_r:] + dYdx_boundary[-N_r:], beta * U)
        loss_B_initial = torch.mean(Y_initial**2)

        loss_B = loss_B_L + loss_B_R + loss_B_initial

        loss = loss_F + loss_B

        if verbose:
            print(
                "Y. loss_F: %.5e, loss_B: %.5e, loss: %.5e" % (loss_F, loss_B, loss),
                end="   ",
            )
        return loss

    def loss_P(self, X_boundary, X_final, X_interior, verbose=False):
        # 计算右边界点的个数
        N_r = torch.sum(X_boundary[:, 0]).detach().cpu().numpy().astype(np.int32)

        P = self.model_P(X_interior)

        dPdX = torch.autograd.grad(
            P,
            X_interior,
            grad_outputs=torch.ones_like(P, device=device),
            retain_graph=True,
            create_graph=True,
        )[0]

        dPdx = dPdX[:, 0:1]
        dPdt = dPdX[:, 1:2]

        dPdxdX = torch.autograd.grad(
            dPdx,
            X_interior,
            grad_outputs=torch.ones_like(dPdx, device=device),
            retain_graph=True,
            create_graph=True,
        )[0]

        dPdxdx = dPdxdX[:, 0:1]

        loss_F = self.criterion(-dPdt, dPdxdx)

        P_boundary = self.model_P(X_boundary)

        dPdX_boundary = torch.autograd.grad(
            P_boundary,
            X_boundary,
            grad_outputs=torch.ones_like(P_boundary, device=device),
            retain_graph=True,
            create_graph=True,
        )[0]

        dPdx_boundary = dPdX_boundary[:, 0:1]

        loss_B_L = torch.mean(dPdx_boundary[:-N_r] ** 2)
        loss_B_R = torch.mean((P_boundary[-N_r:] + dPdx_boundary[-N_r:]) ** 2)

        P_final = self.model_P(X_final)
        Y_final = self.model_Y(X_final)
        Y_d = Y_d_fun(X_final)

        loss_B_initial = self.criterion(P_final, Y_final - Y_d)

        loss_B = loss_B_L + loss_B_R + loss_B_initial

        loss = loss_F + loss_B

        if verbose:
            print(
                "P. loss_F: %.5e, loss_B: %.5e, loss: %.5e" % (loss_F, loss_B, loss),
                end="   ",
            )
        return loss

    def train_YP(self, X_boundary, X_initial, X_final, X_interior, verbose=False):
        self.optimizer_Y.zero_grad()
        self.optimizer_P.zero_grad()
        L_Y = self.loss_Y(X_boundary, X_initial, X_interior, verbose)
        L_P = self.loss_P(X_boundary, X_final, X_interior, verbose)
        L = L_Y + L_P
        L.backward()
        self.optimizer_Y.step()
        self.optimizer_P.step()

        # 是否使用 scheduler
        if self.scheduler_activation:
            self.scheduler_Y.step()
            self.scheduler_P.step()

    def train(self, format="png", verbose=False):
        # 预训练
        X_boundary, X_initial, X_final, X_interior = self.generate_uniform_data(
            self.L_train_warm_n
        )
        for i_warm in range(self.iter_warm):
            # 生成数据
            if (i_warm + 1) % self.display_every == 0:
                print("iter %3d.  " % (i_warm + 1), end="")
                self.train_YP(X_boundary, X_initial, X_final, X_interior, True)
                print("\n", end="")
            else:
                self.train_YP(X_boundary, X_initial, X_final, X_interior, False)

        if verbose:
            self.draw_Y(format=format)
            self.draw_U(format=format)

        for i_epoch in range(self.iter_epoch):
            print("\nepoch %3d" % (i_epoch + 1))
            # 生成数据
            X_boundary, X_initial, X_final, X_interior = self.generate_random_data(
                self.L_train_n
            )

            # 利用 verbose 进行观察
            if verbose:
                for i_finetune in range(self.iter_finetune - 1):
                    if i_finetune % self.display_every == 0:
                        print("iter %3d.  " % (i_finetune + 1), end="")
                        self.train_YP(X_boundary, X_initial, X_final, X_interior, True)
                        print("\n", end="")
                    else:
                        self.train_YP(X_boundary, X_initial, X_final, X_interior, False)
                print("Final.     ", end="")
                self.train_YP(X_boundary, X_initial, X_final, X_interior, True)
                print("\n", end="")
            else:
                for i_finetune in range(self.iter_finetune - 1):
                    self.train_YP(X_boundary, X_initial, X_final, X_interior, False)
                print("Final.     ", end="")
                self.train_YP(X_boundary, X_initial, X_final, X_interior, True)
                print("\n", end="")

        self.draw_Y(format=format)
        self.draw_U(format=format)

    def draw_Y(self, format="png", save=True):
        N = 200
        temp_x = np.linspace(0, 1, N, dtype=dtype)
        temp_t = np.linspace(0, time_T, N, dtype=dtype)

        x, t = np.meshgrid(temp_x, temp_t)

        temp_xy = np.dstack((x, t)).reshape(-1, 2)
        X = torch.tensor(temp_xy, device=device)

        with torch.no_grad():
            Y = self.model_Y(X).reshape(N, N).cpu().numpy()

        fig = plt.figure(figsize=plt.figaspect(0.45))

        # 3D 绘图
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(x, t, Y, cmap=cm.jet)
        ax1.set(xlabel="x", ylabel="t", zlabel="y")

        # colormesh
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_aspect("equal")
        cs = ax2.pcolormesh(x, t, Y, cmap=cm.jet)
        ax2.set(xlabel="x", ylabel="t", title="state")
        colorbar(cs)

        plt.show()

        if save:
            s = (
                time.strftime("./figures/state-%d-%H-%M-%S", time.localtime())
                + "."
                + format
            )
            plt.savefig(s)

        plt.close()

        # 画出最终时刻 t = T 的 state
        fig3, ax3 = plt.subplots()
        ax3.plot(temp_x, Y[-1:, :].squeeze())
        ax3.set(xlabel="x", ylabel="state")

        if save:
            s = (
                time.strftime("./figures/state_final-%d-%H-%M-%S", time.localtime())
                + "."
                + format
            )
            plt.savefig(s)

        plt.close()

    def draw_U(self, format="png", save=True):
        N = 200
        temp_t = np.linspace(0.0, time_T, N, dtype=dtype).reshape(-1, 1)
        temp_xt = np.hstack(
            (np.full((N, 1), time_T, dtype=dtype).reshape(-1, 1), temp_t)
        )
        T = torch.tensor(temp_xt, device=device)

        with torch.no_grad():
            U = self.model_U(T).cpu().numpy()

        fig, ax = plt.subplots()
        ax.plot(temp_t, U)
        ax.set(xlabel="t", ylabel="u")

        plt.show()

        if save:
            s = (
                time.strftime("./figures/control-%d-%H-%M-%S", time.localtime())
                + "."
                + format
            )
            plt.savefig(s)

        plt.close()

    def generate_uniform_data(self, train_n: dict):
        X_boundary = self.geomtime.uniform_boundary_points(train_n["boundary"]).astype(
            dtype=dtype
        )
        X_boundary = X_boundary[np.argsort(X_boundary[:, 0])]

        X_x = self.geomtime.geometry.uniform_points(train_n["initial"]).astype(
            dtype=dtype
        )

        t_initial = self.geomtime.timedomain.t0
        t_final = self.geomtime.timedomain.t1

        X_initial = np.hstack(
            (X_x, np.full([train_n["initial"], 1], t_initial, dtype=dtype))
        )
        X_final = np.hstack(
            (X_x, np.full([train_n["initial"], 1], t_final, dtype=dtype))
        )

        X_interior = self.geomtime.uniform_points(
            train_n["interior"], boundary=False
        ).astype(dtype=dtype)
        return (
            torch.tensor(X_boundary, device=device, requires_grad=True),
            torch.tensor(X_initial, device=device, requires_grad=True),
            torch.tensor(X_final, device=device, requires_grad=True),
            torch.tensor(X_interior, device=device, requires_grad=True),
        )

    def generate_random_data(self, train_n: dict):
        X_boundary = self.geomtime.random_boundary_points(train_n["boundary"]).astype(
            dtype=dtype
        )
        X_boundary = X_boundary[np.argsort(X_boundary[:, 0])]

        X_x = self.geomtime.geometry.random_points(train_n["initial"]).astype(
            dtype=dtype
        )

        t_initial = self.geomtime.timedomain.t0
        t_final = self.geomtime.timedomain.t1

        X_initial = np.hstack(
            (X_x, np.full([train_n["initial"], 1], t_initial, dtype=dtype))
        )
        X_final = np.hstack(
            (X_x, np.full([train_n["initial"], 1], t_final, dtype=dtype))
        )

        X_interior = self.geomtime.random_points(train_n["interior"]).astype(
            dtype=dtype
        )
        return (
            torch.tensor(X_boundary, device=device, requires_grad=True),
            torch.tensor(X_initial, device=device, requires_grad=True),
            torch.tensor(X_final, device=device, requires_grad=True),
            torch.tensor(X_interior, device=device, requires_grad=True),
        )

    def save_model(self, name):
        # 创建文件夹存储图片
        model_path = "./model/" + name + "/"
        # 判断是否存在文件夹如果不存在则创建为文件夹
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.save(self.model_P.state_dict(), model_path + "model_P.pt")
        torch.save(self.model_Y.state_dict(), model_path + "model_Y.pt")

    def load_model(self, name):
        model_path = "./model/" + name + "/"

        model_P_state_dict = torch.load(model_path + "model_P.pt")
        model_Y_state_dict = torch.load(model_path + "model_Y.pt")

        self.model_P.load_state_dict(model_P_state_dict)
        self.model_Y.load_state_dict(model_Y_state_dict)


bipinn = Bangbang()
bipinn.train(verbose=True)
# bipinn.save_model('bang')
