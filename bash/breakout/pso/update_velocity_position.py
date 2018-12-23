import numpy as np

pos = np.loadtxt("particle_pos.txt")
vel = np.loadtxt("particle_vel.txt")
best = np.loadtxt("particle_best.txt")
omega = 1/(2 * np.log(2))
c = 1/2 + np.log(2)

g_pos = pos[0]
g = 0

num_dim = 5

df = pd.read_csv('results.csv', index_col=0)

for index, x in enumerate(pos):
    avg_reward = 0
    for seed in range(5):
        model_name = '_'.join([str(x) for x in [seed, x[0], x[1], x[2], x[3], x[4]]])
        avg_reward += df.loc[model_name]['avg. reward']
    perf = np.mean(avg_reward)
    if best[index][num_dim] < np.mean(perf):
        best[index] = [x[0], x[1], x[2], x[3], x[4], perf]
    if g < best[index][num_dim]:
        g = best[index][num_dim]
        g_pos = [best[index][0], best[index][1], best[index][2], best[index][3], best[index][4]]

new_vel = np.zeros_like(vel)
new_pos = np.zeros_like(pos)
for index, x in enumerate(pos):
    for dim in range(num_dim):
        new_vel[index][dim] = omega * vel[index][dim] + (np.random.random_sample() * c  * (best[index][dim] - x[dim])) + (np.random.random_sample() * c  * (g_pos[dim] - x[dim]))
        new_pos[index][dim] = x[dim] + new_vel[index][dim]

np.savetxt("particle_pos.txt", new_pos)
np.savetxt("particle_vel.txt", new_vel)
np.savetxt("particle_best.txt", best)