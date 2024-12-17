import matplotlib.pyplot as plt


def f(x, y):
    return (2 - x) ** 2 + 10 * (y ** 2 - x) ** 2


class PsoTwoVar:
    def __init__(self, x_vals, y_vals, v_vals, c_vals, r_vals, w_val):
        self.x = x_vals
        self.y = y_vals
        self.vx = v_vals
        self.vy = v_vals
        self.c = c_vals
        self.r = r_vals
        self.w = w_val
        self.old_x = self.x.copy()
        self.old_y = self.y.copy()
        self.p_best_x = self.x.copy()
        self.p_best_y = self.y.copy()
        self.g_best_x = 0.0
        self.g_best_y = 0.0

    def find_p_best(self):
        for i in range(len(self.x)):
            if f(self.x[i], self.y[i]) < f(self.p_best_x[i], self.p_best_y[i]):
                self.p_best_x[i] = self.x[i]
                self.p_best_y[i] = self.y[i]
            else:
                self.p_best_x[i] = self.old_x[i]
                self.p_best_y[i] = self.old_y[i]

    def find_g_best(self):
        min_val = f(self.x[0], self.y[0])
        min_index = 0
        for i in range(1, len(self.x)):
            fx = f(self.x[i], self.y[i])
            if fx < min_val:
                min_val = fx
                min_index = i
        self.g_best_x = self.x[min_index]
        self.g_best_y = self.y[min_index]

    def update_v(self):
        for i in range(len(self.x)):
            self.vx[i] = (self.w * self.vx[i]) + (self.c[0] * self.r[0] * (self.p_best_x[i] -
                                                                           self.x[i])) + (self.c[1] * self.r[1] * (self.g_best_x - self.x[i]))
            self.vy[i] = (self.w * self.vy[i]) + (self.c[0] * self.r[0] * (self.p_best_y[i] -
                                                                           self.y[i])) + (self.c[1] * self.r[1] * (self.g_best_y - self.y[i]))

    def update_x_and_y(self):
        for i in range(len(self.x)):
            self.old_x[i] = self.x[i]
            self.old_y[i] = self.y[i]
            self.x[i] += self.vx[i]
            self.y[i] += self.vy[i]

    def iter(self, n):
        print("Inisialisasi")
        print("x =", [round(val, 5) for val in self.x])
        print("y =", [round(val, 5) for val in self.y])
        print("vx =", [round(val, 5) for val in self.vx])
        print("vy =", [round(val, 5) for val in self.vy], "\n")
        for i in range(n):
            print("Iterasi ke-", i + 1)
            print("==================")
            print("x =", [round(val, 5) for val in self.x])
            print("y =", [round(val, 5) for val in self.y])
            print("vx =", [round(val, 5) for val in self.vx])
            print("vy =", [round(val, 5) for val in self.vy])
            print("f(x, y) =", [round(f(self.x[j], self.y[j]), 5)
                  for j in range(len(self.x))])
            self.find_p_best()
            self.find_g_best()
            print("Personal Best X =", [round(val, 5)
                  for val in self.p_best_x])
            print("Personal Best Y =", [round(val, 5)
                  for val in self.p_best_y])
            print("Global Best X =", round(self.g_best_x, 5))
            print("Global Best Y =", round(self.g_best_y, 5), "\n")
            self.update_v()
            self.update_x_and_y()
            print("Updated x =", [round(val, 5) for val in self.x])
            print("Updated y =", [round(val, 5) for val in self.y])
            print("Updated vx =", [round(val, 5) for val in self.vx])
            print("Updated vy =", [round(val, 5) for val in self.vy], "\n")

    def plot_iteration(self, n):
        x_values = []
        y_values = []
        vx_values = []
        vy_values = []
        p_best_x_values = []
        p_best_y_values = []
        g_best_x_values = []
        g_best_y_values = []
        fx_y_values = []  # Menyimpan nilai f(x, y)

        for i in range(n):
            x_values.append(self.x.copy())
            y_values.append(self.y.copy())
            vx_values.append(self.vx.copy())
            vy_values.append(self.vy.copy())
            p_best_x_values.append(self.p_best_x.copy())
            p_best_y_values.append(self.p_best_y.copy())
            g_best_x_values.append([self.g_best_x] * len(self.x))
            g_best_y_values.append([self.g_best_y] * len(self.y))

            # Menghitung f(x, y) pada setiap iterasi
            fx_y_values.append([f(self.x[j], self.y[j])
                               for j in range(len(self.x))])

            self.find_p_best()
            self.find_g_best()
            self.update_v()
            self.update_x_and_y()

        iterations = list(range(1, n + 1))

        fig, axs = plt.subplots(2, 4, figsize=(16, 8))

        axs[0, 0].plot(iterations, x_values)
        axs[0, 0].set_title('x per Iterasi')
        axs[0, 0].set_xlabel('Iterasi')
        axs[0, 0].set_ylabel('Nilai x')
        axs[0, 0].legend([f'Particle {j+1}' for j in range(len(x_values[0]))])
        axs[0, 0].grid(True)

        axs[0, 1].plot(iterations, y_values)
        axs[0, 1].set_title('y per Iterasi')
        axs[0, 1].set_xlabel('Iterasi')
        axs[0, 1].set_ylabel('Nilai y')
        axs[0, 1].legend([f'Particle {j+1}' for j in range(len(y_values[0]))])
        axs[0, 1].grid(True)

        axs[0, 2].plot(iterations, vx_values)
        axs[0, 2].set_title('vx per Iterasi')
        axs[0, 2].set_xlabel('Iterasi')
        axs[0, 2].set_ylabel('Nilai vx')
        axs[0, 2].legend([f'Particle {j+1}' for j in range(len(vx_values[0]))])
        axs[0, 2].grid(True)

        axs[0, 3].plot(iterations, vy_values)
        axs[0, 3].set_title('vy per Iterasi')
        axs[0, 3].set_xlabel('Iterasi')
        axs[0, 3].set_ylabel('Nilai vy')
        axs[0, 3].legend([f'Particle {j+1}' for j in range(len(vy_values[0]))])
        axs[0, 3].grid(True)

        axs[1, 0].plot(iterations, p_best_x_values)
        axs[1, 0].set_title('Personal Best X per Iterasi')
        axs[1, 0].set_xlabel('Iterasi')
        axs[1, 0].set_ylabel('Nilai Personal Best X')
        axs[1, 0].legend(
            [f'Particle {j+1}' for j in range(len(p_best_x_values[0]))])
        axs[1, 0].grid(True)
        axs[1, 1].plot(iterations, p_best_y_values)
        axs[1, 1].set_title('Personal Best Y per Iterasi')
        axs[1, 1].set_xlabel('Iterasi')
        axs[1, 1].set_ylabel('Nilai Personal Best Y')
        axs[1, 1].legend(
            [f'Particle {j+1}' for j in range(len(p_best_y_values[0]))])
        axs[1, 1].grid(True)

        axs[1, 2].plot(iterations, g_best_x_values)
        axs[1, 2].set_title('Global Best X per Iterasi')
        axs[1, 2].set_xlabel('Iterasi')
        axs[1, 2].set_ylabel('Nilai Global Best X')
        axs[1, 2].legend(['Global Best'])
        axs[1, 2].grid(True)

        axs[1, 3].plot(iterations, g_best_y_values)
        axs[1, 3].set_title('Global Best Y per Iterasi')
        axs[1, 3].set_xlabel('Iterasi')
        axs[1, 3].set_ylabel('Nilai Global Best Y')
        axs[1, 3].legend(['Global Best'])
        axs[1, 3].grid(True)

        plt.tight_layout()
        plt.show()

        fig, ax_fxy = plt.subplots(figsize=(8, 6))
        for i in range(len(fx_y_values[0])):
            ax_fxy.plot(iterations, [fx[i] for fx in fx_y_values],
                        label=f'f(x, y) Particle {i+1}')

        ax_fxy.set_title('f(x, y) per Iterasi')
        ax_fxy.set_xlabel('Iterasi')
        ax_fxy.set_ylabel('Nilai f(x, y)')
        ax_fxy.legend()
        ax_fxy.grid(True)

        plt.tight_layout()
        plt.show()