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
omega = 5e-2
time_T = 1.0


def Y_d_fun(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    t = X[:, 2:3]
    return (
        torch.pow(x, 2)
        * (3 - 2 * x)
        * torch.pow(y, 2)
        * (3 - 2 * y)
        * torch.sin(0.5 * torch.pi * t)
    )


class MyRectangle(geometry.geometry_nd.Hypercube):
    """
    Args:
        xmin: Coordinate of bottom left corner.
        xmax: Coordinate of top right corner.
    """

    def __init__(self, xmin, xmax):
        super().__init__(xmin, xmax)
        self.perimeter = 2 * np.sum(self.xmax - self.xmin)
        self.area = np.prod(self.xmax - self.xmin)

    def uniform_boundary_points(self, n):
        nx, ny = np.ceil(n / self.perimeter * (self.xmax - self.xmin)).astype(int)
        xbot = np.hstack(
            (
                np.linspace(self.xmin[0], self.xmax[0], num=nx, endpoint=False)[
                    :, None
                ],
                np.full([nx, 1], self.xmin[1]),
            )
        )
        yrig = np.hstack(
            (
                np.full([ny, 1], self.xmax[0]),
                np.linspace(self.xmin[1], self.xmax[1], num=ny, endpoint=False)[
                    :, None
                ],
            )
        )
        xtop = np.hstack(
            (
                np.linspace(self.xmin[0], self.xmax[0], num=nx + 1)[1:, None],
                np.full([nx, 1], self.xmax[1]),
            )
        )
        ylef = np.hstack(
            (
                np.full([ny, 1], self.xmin[0]),
                np.linspace(self.xmin[1], self.xmax[1], num=ny + 1)[1:, None],
            )
        )
        x = {
            "bottom": np.vstack(xbot),
            "top": np.vstack(xtop),
            "left": np.vstack(ylef),
            "right": np.vstack(yrig),
        }
        # if n != len(x):
        #     print(
        #         "Warning: {} points required, but {} points sampled.".format(
        #             n, len(x))
        #     )
        return x

    def random_boundary_points(self, n, random="pseudo"):
        l1 = self.xmax[0] - self.xmin[0]
        l2 = l1 + self.xmax[1] - self.xmin[1]
        l3 = l2 + l1
        u = np.ravel(geometry.sampler.sample(n + 2, 1, random))
        # Remove the possible points very close to the corners
        u = u[np.logical_not(np.isclose(u, l1 / self.perimeter))]
        u = u[np.logical_not(np.isclose(u, l3 / self.perimeter))]
        u = u[:n]

        u *= self.perimeter
        xl = []
        xt = []
        xr = []
        xb = []
        for l in u:
            if l < l1:
                xb.append([self.xmin[0] + l, self.xmin[1]])
            elif l < l2:
                xr.append([self.xmax[0], self.xmin[1] + l - l1])
            elif l < l3:
                xt.append([self.xmax[0] - l + l2, self.xmax[1]])
            else:
                xl.append([self.xmin[0], self.xmax[1] - l + l3])
        x = {
            "left": np.vstack(xl),
            "right": np.vstack(xr),
            "bottom": np.vstack(xb),
            "top": np.vstack(xt),
        }
        return x

    @staticmethod
    def is_valid(vertices):
        """Check if the geometry is a Rectangle."""
        return (
            len(vertices) == 4
            and np.isclose(np.prod(vertices[1] - vertices[0]), 0)
            and np.isclose(np.prod(vertices[2] - vertices[1]), 0)
            and np.isclose(np.prod(vertices[3] - vertices[2]), 0)
            and np.isclose(np.prod(vertices[0] - vertices[3]), 0)
        )


class MyGeometryXTime(geometry.GeometryXTime):
    def __init__(self, geometry, timedomain):
        super().__init__(geometry, timedomain)

    def uniform_boundary_points(self, n):
        """Uniform boundary points on the spatio-temporal domain.

        Geometry surface area ~ bbox.
        Time surface area ~ diam.
        """
        if self.geometry.dim == 1:
            nx = 2
        else:
            s = 2 * sum(
                map(
                    lambda l: l[0] * l[1],
                    itertools.combinations(
                        self.geometry.bbox[1] - self.geometry.bbox[0], 2
                    ),
                )
            )
            nx = int((n * s / self.timedomain.diam) ** 0.5)
        nt = int(np.ceil(n / nx))
        x = self.geometry.uniform_boundary_points(nx)
        t = np.linspace(
            self.timedomain.t1,
            self.timedomain.t0,
            num=nt,
            endpoint=False,
            dtype=dtype,
        )

        xt_dict = {}
        for key, value in x.items():
            xt = []
            nx = len(x[key])
            for ti in t:
                xt.append(np.hstack((value, np.full([nx, 1], ti))))
            xt = np.vstack(xt)
            xt_dict[key] = xt.astype(dtype=dtype)
        return xt_dict

    def random_boundary_points(self, n, random="pseudo"):
        x = self.geometry.random_boundary_points(n, random=random)

        xt_dict = {}
        for key, value in x.items():
            nx = len(value)
            t = self.timedomain.random_points(nx, random=random)
            xt_dict[key] = np.hstack((value, t)).astype(dtype=dtype)
        return xt_dict

    def generate_uniform_data(self, train_n: dict):
        X_boundary = self.uniform_boundary_points(train_n["boundary"])

        X_x = self.geometry.uniform_points(train_n["initial"]).astype(dtype=dtype)

        t_initial = self.timedomain.t0
        t_final = self.timedomain.t1

        X_initial = np.hstack(
            (X_x, np.full([train_n["initial"], 1], t_initial, dtype=dtype))
        )
        X_final = np.hstack(
            (X_x, np.full([train_n["initial"], 1], t_final, dtype=dtype))
        )

        X_interior = self.uniform_points(train_n["interior"], boundary=False).astype(
            dtype=dtype
        )

        X_boundary_train = {}
        for key, value in X_boundary.items():
            X_boundary_train[key] = torch.tensor(
                value, device=device, requires_grad=True
            )

        return (
            X_boundary_train,
            torch.tensor(X_initial, device=device, requires_grad=True),
            torch.tensor(X_final, device=device, requires_grad=True),
            torch.tensor(X_interior, device=device, requires_grad=True),
        )

    def generate_random_data(self, train_n: dict):
        X_boundary = self.random_boundary_points(train_n["boundary"])

        X_x = self.geometry.random_points(train_n["initial"]).astype(dtype=dtype)

        t_initial = self.timedomain.t0
        t_final = self.timedomain.t1

        X_initial = np.hstack(
            (X_x, np.full([train_n["initial"], 1], t_initial, dtype=dtype))
        )
        X_final = np.hstack(
            (X_x, np.full([train_n["initial"], 1], t_final, dtype=dtype))
        )

        X_interior = self.random_points(train_n["interior"]).astype(dtype=dtype)

        X_boundary_train = {}
        for key, value in X_boundary.items():
            X_boundary_train[key] = torch.tensor(
                value, device=device, requires_grad=True
            )

        return (
            X_boundary_train,
            torch.tensor(X_initial, device=device, requires_grad=True),
            torch.tensor(X_final, device=device, requires_grad=True),
            torch.tensor(X_interior, device=device, requires_grad=True),
        )


class Heat:
    def __init__(self):

        self.learning_rate_Y = 1e-3
        self.learning_rate_P = 1e-3

        self.Y_dim = 3
        self.P_dim = 3

        self.iter_warm = 500
        self.iter_epoch = 800
        self.iter_finetune = 50
        self.display_every = 20

        self.L_train_warm_n = {
            "interior": 50**3,
            "boundary": 4 * 50 * 50,
            "initial": 50**2,
        }

        self.L_train_n = {
            "interior": 20**3,
            "boundary": 4 * 20 * 20,
            "initial": 20**2,
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

        self.geomtime = MyGeometryXTime(
            MyRectangle((0, 0), (1, 1)), geometry.TimeDomain(0, time_T)
        )

    def model_U(self, X):
        P = self.model_P(X)
        U = -P / omega
        return U

    def loss_Y(self, X_boundary, X_initial, X_interior, verbose=False):

        Y = self.model_Y(X_interior)
        U = self.model_U(X_interior)

        # 计算 pde 损失：loss_F
        dYdX = torch.autograd.grad(
            Y,
            X_interior,
            grad_outputs=torch.ones_like(Y, device=device),
            retain_graph=True,
            create_graph=True,
        )[0]

        dYdx = dYdX[:, 0:1]
        dYdy = dYdX[:, 1:2]
        dYdt = dYdX[:, 2:3]

        dYdxdX = torch.autograd.grad(
            dYdx,
            X_interior,
            grad_outputs=torch.ones_like(dYdx, device=device),
            retain_graph=True,
            create_graph=True,
        )[0]
        dYdydX = torch.autograd.grad(
            dYdy,
            X_interior,
            grad_outputs=torch.ones_like(dYdy, device=device),
            retain_graph=True,
            create_graph=True,
        )[0]
        dYdxdx = dYdxdX[:, 0:1]
        dYdydy = dYdydX[:, 1:2]

        loss_F = self.criterion(dYdt - (dYdxdx + dYdydy), U)

        # 计算边界损失：loss_B
        Y_boundary = {}
        dYdX_boundary = {}
        for key, value in X_boundary.items():
            Y_boundary[key] = self.model_Y(value)
            dYdX_boundary[key] = torch.autograd.grad(
                Y_boundary[key],
                value,
                grad_outputs=torch.ones_like(Y_boundary[key], device=device),
                retain_graph=True,
                create_graph=True,
            )[0]

        dYdX_boundary = torch.vstack(
            (
                dYdX_boundary["left"][:, 0:1],
                dYdX_boundary["right"][:, 0:1],
                dYdX_boundary["bottom"][:, 1:2],
                dYdX_boundary["top"][:, 1:2],
            )
        )

        Y_initial = self.model_Y(X_initial)

        loss_B = torch.mean(dYdX_boundary**2) * 4 + torch.mean(Y_initial**2)

        loss = loss_F + loss_B

        if verbose:
            print(
                "Y. loss_F: %.5e, loss_B: %.5e, loss: %.5e" % (loss_F, loss_B, loss),
                end="   ",
            )
        return loss

    def loss_P(self, X_boundary, X_final, X_interior, verbose=False):

        P = self.model_P(X_interior)
        Y = self.model_Y(X_interior)
        Y_d = Y_d_fun(X_interior)

        # 计算 pde 损失：loss_F
        dPdX = torch.autograd.grad(
            P,
            X_interior,
            grad_outputs=torch.ones_like(P, device=device),
            retain_graph=True,
            create_graph=True,
        )[0]

        dPdx = dPdX[:, 0:1]
        dPdy = dPdX[:, 1:2]
        dPdt = dPdX[:, 2:3]

        dPdxdX = torch.autograd.grad(
            dPdx,
            X_interior,
            grad_outputs=torch.ones_like(dPdx, device=device),
            retain_graph=True,
            create_graph=True,
        )[0]
        dPdydX = torch.autograd.grad(
            dPdy,
            X_interior,
            grad_outputs=torch.ones_like(dPdy, device=device),
            retain_graph=True,
            create_graph=True,
        )[0]
        dPdxdx = dPdxdX[:, 0:1]
        dPdydy = dPdydX[:, 1:2]

        loss_F = self.criterion(-dPdt - dPdxdx - dPdydy, Y - Y_d)

        # 计算边界损失：loss_B
        P_boundary = {}
        dPdX_boundary = {}
        for key, value in X_boundary.items():
            P_boundary[key] = self.model_P(value)
            dPdX_boundary[key] = torch.autograd.grad(
                P_boundary[key],
                value,
                grad_outputs=torch.ones_like(P_boundary[key], device=device),
                retain_graph=True,
                create_graph=True,
            )[0]

        dPdX_boundary = torch.vstack(
            (
                dPdX_boundary["left"][:, 0:1],
                dPdX_boundary["right"][:, 0:1],
                dPdX_boundary["bottom"][:, 1:2],
                dPdX_boundary["top"][:, 1:2],
            )
        )

        P_final = self.model_P(X_final)

        loss_B = torch.mean(dPdX_boundary**2) * 4 + torch.mean(P_final**2)

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
        X_boundary, X_initial, X_final, X_interior = (
            self.geomtime.generate_uniform_data(self.L_train_warm_n)
        )
        for i_warm in range(self.iter_warm):
            # 生成数据
            if (i_warm + 1) % self.display_every == 0:
                print("iter %3d.  " % (i_warm + 1), end="")
                self.train_YP(X_boundary, X_initial, X_final, X_interior, True)
                print("\n", end="")
            else:
                self.train_YP(X_boundary, X_initial, X_final, X_interior, False)

        self.draw("y_d", format=format)
        if verbose:
            for var in ["y", "u", "p"]:
                self.draw(var, format=format)

        for i_epoch in range(self.iter_epoch):
            print("\nepoch %3d" % (i_epoch + 1))
            # 生成数据
            X_boundary, X_initial, X_final, X_interior = (
                self.geomtime.generate_random_data(self.L_train_n)
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

    def plot_data(self, func, t):
        # 设置画图相关参数
        N = 200
        temp_x = np.linspace(0.0, 1.0, N, dtype=dtype)
        temp_y = np.linspace(0.0, 1.0, N, dtype=dtype)
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

    def draw(self, var, t=[time_T / 2, time_T], format="png", save=True):

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
            print("Can't plot variable %s" % name)

        x, y, Y = self.plot_data(func, t)

        N_t = len(t)
        fig, ax = plt.subplots(1, N_t)
        fig.subplots_adjust(wspace=0.6)

        # 画 t 时刻的 state
        for i in range(N_t):
            ax[i].set_aspect("equal")
            cs = ax[i].pcolormesh(x, y, Y[i], cmap=cm.hot)
            ax[i].set(xlabel="x", ylabel="y", title=f"{name} when t = {t[i]}")
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
        np.savetxt("heat-loss.dat", data)

    def save_data(self, var, t=[time_T / 2, time_T]):

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

        # 设置画图相关参数
        N = 201
        temp_x = np.linspace(0.0, 1.0, N, dtype=dtype)
        temp_y = np.linspace(0.0, 1.0, N, dtype=dtype)
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

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        for i in range(len(t)):
            temp_Y = Y[i].reshape(-1, 1)
            data = np.hstack((x, y, temp_Y))

            np.savetxt("heat2d-t%02d-%s.dat" % (t[i] * 1e2, name), data)


def write_tex(write_path, model_name, var, t_array):
    if var == "y":
        var = "state"
    elif var == "u":
        var = "control"
    elif var == "error":
        var = "state_error"

    data_path = os.getcwd().replace("\\", "/")

    for t in t_array:

        # heat2d-t025-state
        file = "%s-t%02d-%s" % (model_name, t * 1e2, var)

        file_name = "{" + data_path + "/" + file + ".dat" + "}"

        write_file = write_path + file + ".tex"

        tex = (
            "\documentclass{myplot}"
            + "\n\n"
            + "\\begin{document}\n"
            + "\t\\myplotthreed{x_1}{x_2}{hot}"
            + file_name
            + "\n"
            + "\end{document}\n"
        )

        with open(write_file, "w") as f:
            f.write(tex)
            f.close()


bipinn = Heat()
bipinn.train(verbose=False, format="pdf")
bipinn.save_model("heat")
bipinn.save_loss()

bipinn.load_model("heat")

t_array = [0.25, 0.75]

for var in ["y", "u", "error"]:
    bipinn.save_data(var, t_array)
    bipinn.draw(var, t_array)
    write_tex("../../sysuthesis/tikz/", "heat2d", var, t_array)
