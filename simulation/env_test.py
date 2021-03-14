from env.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
 
env = KukaDiverseObjectEnv(renders=True, isDiscrete=False)
env.reset()
 
for t in range(30):
    action = env.action_space.sample()
    next_obs, reward, done, _ = env.step(action)
 
    if done == True:
        break
    obs = next_obs