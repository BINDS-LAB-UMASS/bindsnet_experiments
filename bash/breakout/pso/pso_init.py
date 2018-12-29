import numpy as np

num_dim = 5
num_particles = int(10 + 2 * np.sqrt(num_dim))

particle_positions = np.zeros([num_particles, num_dim])
particle_velocities = np.zeros([num_particles, num_dim])
best_positions = np.zeros([num_particles, num_dim + 1])
for i in range(num_particles):
    for dim in range(num_dim):
        particle_positions[i][dim] = np.random.random_sample()*500
        particle_velocities[i][dim] = (np.random.random_sample() * 500 - particle_positions[i][dim])/2
        best_positions[i][dim] = particle_positions[i][dim]
    best_positions[i][num_dim] = 0

np.savetxt("particle_pos.txt", particle_positions)
np.savetxt("particle_vel.txt", particle_velocities)
np.savetxt("particle_best.txt", best_positions)