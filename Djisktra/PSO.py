import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def f(x, y):
    return np.cos(2 * x + y) + (2 *x - y) ** 2 - 1.5 * x + 2.5 * y +1

class PSO:
    def __init__(self, position_x, position_y, velocity_x, velocity_y, w, c1, c2, r1, r2, particle_id):
        self.particle_id = particle_id
        self.position_x = position_x
        self.position_y = position_y
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.r1 = r1
        self.r2 = r2
        self.pBest_position_x = position_x.copy()
        self.pBest_position_y = position_y.copy()
        self.pBest_value = f(position_x[0], position_y[0])
        self.gBestx = 0
        self.gBesty = 0
        self.f_xy_values = []

    def update_Pbest(self):
        for i in range(len(self.position_x)):  
            particle_value = f(self.position_x[i], self.position_y[i])  
            if particle_value < self.pBest_value:
                self.pBest_position_x[i] = self.position_x[i]
                self.pBest_position_y[i] = self.position_y[i]
                self.pBest_value = particle_value  

    def update_gBest(self, particles):
        fValues = [f(p.pBest_position_x[0], p.pBest_position_y[0]) for p in particles]
        minIndex = np.argmin(fValues)
        
        # Set global best untuk semua partikel
        global_best_particle = particles[minIndex]
        for particle in particles:
            particle.gBestx = global_best_particle.pBest_position_x[0]
            particle.gBesty = global_best_particle.pBest_position_y[0]

    def update_velocity(self):
        for i in range(len(self.position_x)):
            self.velocity_x[i] = (
                (self.w * self.velocity_x[i])
                + (self.c1 * self.r1 * (self.pBest_position_x[i] - self.position_x[i]))
                + (self.c2 * self.r2 * (self.gBestx - self.position_x[i]))
            )
            self.velocity_y[i] = (
                (self.w * self.velocity_y[i])
                + (self.c1 * self.r1 * (self.pBest_position_y[i] - self.position_y[i]))
                + (self.c2 * self.r2 * (self.gBesty - self.position_y[i]))
            )
    
    def update_position(self):
        for i in range(len(self.position_x)):
            self.position_x[i] = self.position_x[i] + self.velocity_x[i]
            self.position_y[i] = self.position_y[i] + self.velocity_y[i]
            
    def boundary_handling(self, lower_bound, upper_bound):
        for i in range(len(self.position_x)):
            # Periksa dan perbaiki posisi x
            if self.position_x[i] < lower_bound[0]:
                self.position_x[i] = lower_bound[0]
            elif self.position_x[i] > upper_bound[0]:
                self.position_x[i] = upper_bound[0]

            # Periksa dan perbaiki posisi y
            if self.position_y[i] < lower_bound[1]:
                self.position_y[i] = lower_bound[1]
            elif self.position_y[i] > upper_bound[1]:
                self.position_y[i] = upper_bound[1]


    def iterate_pso_plot(self, particles, n_iterations, lower_bound, upper_bound):
        x_position = [[] for _ in range(len(particles))]
        y_position = [[] for _ in range(len(particles))]
        x_velocity = [[] for _ in range(len(particles))]
        y_velocity = [[] for _ in range(len(particles))]
        p_best_x = [[] for _ in range(len(particles))]
        p_best_y = [[] for _ in range(len(particles))]
        g_best_x = []
        g_best_y = []
        g_best_values = []
        f_xy_values = [[] for _ in range(len(particles))]  

        
        for iteration in range(n_iterations):
            # print(f"\n{'='*20} ITERASI {iteration + 1} {'='*20}")
            
            for i, p in enumerate(particles):
                x_position[i].append((p.position_x[0]))
                y_position[i].append((p.position_y[0]))
                x_velocity[i].append((p.velocity_x[0]))
                y_velocity[i].append((p.velocity_y[0]))
                p_best_x[i].append((p.pBest_position_x[0]))
                p_best_y[i].append((p.pBest_position_y[0]))
                
                f_xy_values[i].append(f(p.position_x[0], p.position_y[0]))
                

            for particle in particles:
                particle.update_Pbest()
                
            particles[0].update_gBest(particles)
            global_best_x = particles[0].gBestx
            global_best_y = particles[0].gBesty
            g_best_values.append(f(global_best_x, global_best_y))
            g_best_x.append(global_best_x)
            g_best_y.append(global_best_y)
                

            global_best_x = particles[0].gBestx
            global_best_y = particles[0].gBesty

            for i, particle in enumerate(particles):
                # particle.print_iteration_info(global_best_x, global_best_y)
                particle.update_velocity()
                particle.update_position()
                particle.boundary_handling(lower_bound, upper_bound)
       
            
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        for i, values in enumerate(x_position):
            axs[0, 0].plot(range(n_iterations), values, marker='o', label=f'Partikel {i + 1}')
            axs[0, 0].set_title('x per Iterasi')
            axs[0, 0].set_xlabel('Iterasi')
            axs[0, 0].set_ylabel('Posisi x')
            axs[0, 0].legend()
            axs[0, 0].grid(True)
        for i, values in enumerate(y_position):
            axs[1, 0].plot(range(n_iterations), values)
            axs[1, 0].set_title('y per Iterasi')
            axs[1, 0].set_xlabel('Iterasi')
            axs[1, 0].set_ylabel('Posisi y')
            axs[1, 0].legend([f'Particle {j+1}' for j in range(len(y_position[0]))])
            axs[1, 0].grid(True)
        for i, values in enumerate(x_velocity):
            axs[0, 1].plot(range(n_iterations), values)
            axs[0, 1].set_title('v per Iterasi')
            axs[0, 1].set_xlabel('Iterasi')
            axs[0, 1].set_ylabel('Velocity x')
            axs[0, 1].legend([f'Particle {j+1}' for j in range(len(x_velocity[0]))])
            axs[0, 1].grid(True)
        for i, values in enumerate(y_velocity):
            axs[1, 1].plot(range(n_iterations), values)
            axs[1, 1].set_title('v per Iterasi')
            axs[1, 1].set_xlabel('Iterasi')
            axs[1, 1].set_ylabel('Velocity y')
            axs[1, 1].legend([f'Particle {j+1}' for j in range(len(y_velocity[0]))])
            axs[1, 1].grid(True)
        for i, values in enumerate(p_best_x):
            axs[0, 2].plot(range(n_iterations), values)
            axs[0, 2].set_title('x per Iterasi')
            axs[0, 2].set_xlabel('Iterasi')
            axs[0, 2].set_ylabel('Personal Best x')
            axs[0, 2].legend([f'Particle {j+1}' for j in range(len(p_best_x[0]))])
            axs[0, 2].grid(True)    
        for i, values in enumerate(p_best_y):
            axs[1, 2].plot(range(n_iterations), values)
            axs[1, 2].set_title('x per Iterasi')
            axs[1, 2].set_xlabel('Iterasi')
            axs[1, 2].set_ylabel('Personal Best y')
            axs[1, 2].legend([f'Particle {j+1}' for j in range(len(p_best_y[0]))])
            axs[1, 2].grid(True)
        axs[0, 3].plot(range(n_iterations), g_best_x, label='gBest x', color='red')
        axs[0, 3].set_title('Global Best x per Iterasi')
        axs[0, 3].set_xlabel('Iterasi')
        axs[0, 3].set_ylabel('Global Best x')
        axs[0, 3].grid(True)
        
        axs[1, 3].plot(range(n_iterations), g_best_y, label='gBest y', color='red')
        axs[1, 3].set_title('Global Best y per Iterasi')
        axs[1, 3].set_xlabel('Iterasi')
        axs[1, 3].set_ylabel('Global Best y')
        axs[1, 3].grid(True)
           
        plt.tight_layout()
        plt.show()    
        
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        for i, values in enumerate(f_xy_values):
            
            axs[0].plot(range(n_iterations), values, marker='o', label=f'Partikel {i + 1}')
            
        axs[0].plot(range(n_iterations), g_best_values, marker='x', label='gBest Value', color='red', linestyle='--')
        axs[0].set_title("Perkembangan Nilai f(x, y) selama Iterasi")
        axs[0].set_xlabel("Iterasi")
        axs[0].set_ylabel("f(x, y)")
        axs[0].legend()
        axs[0].grid(True)
        
    
        x = np.linspace(-5,5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        fig = plt.figure("Particle Swarm Optimization")

        ax = fig.add_subplot(1, 1, 1)
        ac = ax.contourf(X, Y, f(X, Y), cmap='viridis')
        fig.colorbar(ac)
        
        for i in range(min(10, len(particles))):
            ax.scatter(x_position[i][-1], y_position[i][-1], label=f'Partikel {i + 1}')
        axs[1].scatter(x_position[i][-1], y_position[i][-1], label=f'Partikel {i + 1}')
        axs[1].scatter(g_best_x[-1], g_best_y[-1], marker='x', label='gBest Value', color='red', linestyle='--')

        axs[1].set_title("Posisi Partikel dalam Ruang x-y selama Iterasi")
        axs[1].set_xlabel("Posisi x")
        axs[1].set_ylabel("Posisi y")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()    

    def print_iteration_info(self, global_best_x, global_best_y):
        print("=" * 50)
        print(f"**Partikel {self.particle_id}**")
        print("**Inisialisasi Partikel**")
        print(f"Posisi awal x = {self.position_x}, y = {self.position_y}")
        print(f"Kecepatan awal x = {self.velocity_x}, y = {self.velocity_y}")
        print(" ")
        
        print(f"pBest value = {self.pBest_value}")
        
        print(f"Global Best Value = {f(global_best_x, global_best_y)}")
        print(f"Posisi Global Best x = {global_best_x}, y = {global_best_y}")

        print(f"Posisi sekarang x = {self.position_x}, y = {self.position_y}")
        print(f"Kecepatan sekarang x = {self.velocity_x}, y = {self.velocity_y}")
        print("-" * 50)
        
# def run_pso_2_a():
#     # Membuat daftar partikel
#     lower_bound = [-5, -5]
#     upper_bound = [5, 5]
    
#     particles = [
#         PSO([1], [1], [0], [0], 1, 1, 0.5, 1, 1, 1),
#         PSO([1], [-1], [0], [0], 1, 1, 0.5, 1, 1, 2),
#         PSO([2], [1], [0], [0], 1, 1, 0.5, 1, 1, 3)
#     ]
    
#     # Jalankan iterasi PSO
#     particles[0].iterate_pso_plot(particles, 3, lower_bound, upper_bound)
    
def run_pso_2_b():
    lower_bound = [-5, -5]
    upper_bound = [5, 5]
    
    particles = [
        PSO([np.random.uniform(lower_bound[0], upper_bound[0])], [np.random.uniform(lower_bound[1], upper_bound[1])], [0], [0], 1, 1, 0.5, np.random.random(), np.random.random(), 1),
        PSO([np.random.uniform(lower_bound[0], upper_bound[0])], [np.random.uniform(lower_bound[1], upper_bound[1])], [0], [0], 1, 1, 0.5, np.random.random(), np.random.random(), 2),
        PSO([np.random.uniform(lower_bound[0], upper_bound[0])], [np.random.uniform(lower_bound[1], upper_bound[1])], [0], [0], 1, 1, 0.5, np.random.random(), np.random.random(), 3),
        PSO([np.random.uniform(lower_bound[0], upper_bound[0])], [np.random.uniform(lower_bound[1], upper_bound[1])], [0], [0], 1, 1, 0.5, np.random.random(), np.random.random(), 4),
        PSO([np.random.uniform(lower_bound[0], upper_bound[0])], [np.random.uniform(lower_bound[1], upper_bound[1])], [0], [0], 1, 1, 0.5, np.random.random(), np.random.random(), 5),
        PSO([np.random.uniform(lower_bound[0], upper_bound[0])], [np.random.uniform(lower_bound[1], upper_bound[1])], [0], [0], 1, 1, 0.5, np.random.random(), np.random.random(), 6),
        PSO([np.random.uniform(lower_bound[0], upper_bound[0])], [np.random.uniform(lower_bound[1], upper_bound[1])], [0], [0], 1, 1, 0.5, np.random.random(), np.random.random(), 7),
        PSO([np.random.uniform(lower_bound[0], upper_bound[0])], [np.random.uniform(lower_bound[1], upper_bound[1])], [0], [0], 1, 1, 0.5, np.random.random(), np.random.random(), 8),
        PSO([np.random.uniform(lower_bound[0], upper_bound[0])], [np.random.uniform(lower_bound[1], upper_bound[1])], [0], [0], 1, 1, 0.5, np.random.random(), np.random.random(), 9),
        PSO([np.random.uniform(lower_bound[0], upper_bound[0])], [np.random.uniform(lower_bound[1], upper_bound[1])], [0], [0], 1, 1, 0.5, np.random.random(), np.random.random(), 10)
    ] 
    
    # Jalankan iterasi PSO
    particles[0].iterate_pso_plot(particles, 250, lower_bound, upper_bound)
    

if __name__ == "__main__":
    run_pso_2_b()