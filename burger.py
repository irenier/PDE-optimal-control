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
nu = 1e-2
omega = 5e-2


def Y_d_fun(X):
    x = X[:, 0:1]
    return torch.where(
        x < 0.5, torch.ones_like(x, device=device), torch.zeros_like(x, device=device)
    )


def Y_initial_fun(X):
    x = X[:, 0:1]
    return torch.where(
        x < 0.5, torch.ones_like(x, device=device), torch.zeros_like(x, device=device)
    )


class Burger:
    def __init__(self):

        self.learning_rate_Y = 1e-3
        self.learning_rate_P = 1e-3

        self.Y_dim = 2
        self.P_dim = 2

        self.iter_warm = 500
        self.iter_epoch = 800
        self.iter_finetune = 50
        self.display_every = 20

        self.L_train_warm_n = {
            "interior": 100 * 100,
            "boundary": 100 * 2,
            "initial": 100,
        }

        self.L_train_n = {
            "interior": 50 * 50,
            "boundary": 50 * 2,
            "initial": 50,
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
            geometry.geometry_1d.Interval(0, 1), geometry.TimeDomain(0, 1)
        )

    def model_U(self, X):
        P = self.model_P(X)
        U = -P / omega
        return U

    def loss_Y(self, X_boundary, X_initial, X_interior, verbose=False):

        Y_boundary = self.model_Y(X_boundary)
        Y_initial = self.model_Y(X_initial)
        Y = self.model_Y(X_interior)
        U = self.model_U(X_interior)

        Y_initial_true = Y_initial_fun(X_initial)

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

        loss_F = self.criterion(dYdt - nu * dYdxdx + Y * dYdx, U)

        loss_B = torch.mean(Y_boundary**2) * 2.0 + self.criterion(
            Y_initial, Y_initial_true
        )

        loss = loss_F + loss_B

        if verbose:
            print(
                "Y. loss_F: %.5e, loss_B: %.5e, loss: %.5e" % (loss_F, loss_B, loss),
                end="   ",
            )
        return loss

    def loss_P(self, X_boundary, X_final, X_interior, verbose=False):

        P_boundary = self.model_P(X_boundary)
        P_final = self.model_P(X_final)
        P = self.model_P(X_interior)
        Y = self.model_Y(X_interior)

        Y_d = Y_d_fun(X_interior)

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

        loss_F = self.criterion(dPdt + nu * dPdxdx + Y * dPdx, Y_d - Y)

        loss_B = torch.mean(P_boundary**2) * 2.0 + torch.mean(P_final**2)

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
            for var in ["y", "u", "p"]:
                self.draw(var, format=format)

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

        for var in ["y", "u", "p"]:
            self.draw(var, format=format)

    def draw(self, var, format="png", save=True):

        if var == "y":
            func = self.model_Y
            name = "state"
        elif var == "u":
            func = self.model_U
            name = "control"
        elif var == "p":
            func = self.model_P
            name = "adjoint_state"
        elif var == "y_d":
            func = Y_d_fun
            name = "y_d"
        else:
            print("Can't plot variable %s" % name)

        N = 200
        temp_x = np.linspace(0, 1, N, dtype=dtype)
        temp_t = np.linspace(0, 1, N, dtype=dtype)

        x, t = np.meshgrid(temp_x, temp_t)

        temp_xt = np.dstack((x, t)).reshape(-1, 2)
        X = torch.tensor(temp_xt, device=device)

        with torch.no_grad():
            Y = func(X).reshape(N, N).cpu().numpy()

        fig = plt.figure(figsize=plt.figaspect(0.45))

        # 3D 绘图
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(x, t, Y, cmap=cm.jet)
        ax1.set(xlabel="x", ylabel="t", zlabel=var)

        # colormesh
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_aspect("equal")
        cs = ax2.pcolormesh(x, t, Y, cmap=cm.jet)
        ax2.set(xlabel="x", ylabel="t", title=name)
        colorbar(cs)

        plt.show()

        if save:
            s = (
                "./figures/"
                + name
                + time.strftime("-%d-%H-%M-%S", time.localtime())
                + "."
                + format
            )
            plt.savefig(s)

        plt.close()

    def generate_uniform_data(self, train_n: dict):
        X_boundary = self.geomtime.uniform_boundary_points(train_n["boundary"]).astype(
            dtype=dtype
        )

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

        model_P_state_dict = torch.load(model_path + "model_P.pt", map_location=device)
        model_Y_state_dict = torch.load(model_path + "model_Y.pt", map_location=device)

        self.model_P.load_state_dict(model_P_state_dict)
        self.model_Y.load_state_dict(model_Y_state_dict)

    def save_loss(self):
        N = np.arange(len(self.L_array)).reshape(-1, 1)
        loss = np.array(self.L_array).reshape(-1, 1)
        data = np.hstack((N, loss))
        np.savetxt("burger-loss.dat", data)

    def save_data(self, var):

        if var == "y":
            func = self.model_Y
            name = "state"
        elif var == "u":
            func = self.model_U
            name = "control"
        elif var == "p":
            func = self.model_P
            name = "adjoint_state"
        elif var == "y_d":
            func = Y_d_fun
            name = "y_d"
        elif var == "error":
            func = lambda X: self.model_Y(X) - Y_d_fun(X)
            name = "state_error"
        else:
            print("Can't save variable %s" % name)
        N = 201
        temp_x = np.linspace(0, 1, N, dtype=dtype)
        temp_t = np.linspace(0, 1, N, dtype=dtype)

        x, t = np.meshgrid(temp_x, temp_t)

        temp_xt = np.dstack((x, t)).reshape(-1, 2)
        X = torch.tensor(temp_xt, device=device)

        with torch.no_grad():
            Y = func(X).reshape(N, N).cpu().numpy()

        x = x.reshape(-1, 1)
        t = t.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        data = np.hstack((x, t, Y))
        np.savetxt("burger-" + name + ".dat", data)


bipinn = Burger()
bipinn.train(verbose=False, format="pdf")
bipinn.save_model("burger")
bipinn.save_loss()

bipinn.load_model("burger")
bipinn.save_data("y")
bipinn.save_data("u")
bipinn.save_data("error")
