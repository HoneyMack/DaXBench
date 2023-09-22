import jax
import jax.numpy as jnp
from daxbench.core.envs import ShapeRopeEnv, UnfoldCloth1Env

# Crreate the environments
# env = ShapeRopeEnv(batch_size=3, seed=1)
# obs, state = env.reset(env.simulator.key)
env = UnfoldCloth1Env(batch_size=3, seed=2)
obs, state = env.reset(env.simulator.key_global)

# Actions to be simulated in each environment
actions = jnp.array(
   [
      [0.4, 0, 0.4, 0.6, 0, 0.6],
      [0.6, 0, 0.6, 0.4, 0, 0.4],
      [0.4, 0, 0.6, 0.6, 0, 0.4],
   ]
)

obs, reward, done, info = env.step_with_render(actions, state)
next_state = info["state"]
imgs = info["img_list"]

print("obs.shape:", obs.shape)
print("reward:", reward)
print("done:", done)
print("info:", info.keys())

#　観測値を画像で出力
import matplotlib.pyplot as plt
plt.imsave("example_cloth_env2.png", imgs[0])