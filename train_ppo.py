import wandb
import numpy as np
from stable_baselines3 import PPO
from grid_env import GridWorldEnv
# from grid_env_new_reward import GridWorldEnv
import torch as th
from map_generator import generate_random_map, is_map_solvable

# run_id = ""

wandb.init(
    project="hybrid_rl_grid"    
    # id=run_id, 
    # resume="allow"
)

env = GridWorldEnv(generate_random_map())

# policy_kwargs = dict(activation_fn=th.nn.ReLU,
#                      net_arch=dict(pi=[64, 32, 32, 12], vf=[64, 32, 32, 12]))

# model = PPO(policy="MlpPolicy",
#             env=env,            
#             learning_rate=1e-3,
#             policy_kwargs=policy_kwargs,
#             gamma=0.95)

model = PPO.load("models/hybrid_ppo_grid_last", env=env)

desired_avg_reward = 1000

last_score = -10000

try:
    while True:
        # print("New map is generated!")
        grid_map = generate_random_map()
        env = GridWorldEnv(grid_map)
        obs = env.reset()
        solvable, path = is_map_solvable(grid_map, env.agent_position, env.goal_position)
        while not solvable:
            # print("Map not solvable yet!")
            grid_map = generate_random_map()
            env = GridWorldEnv(grid_map)
            obs = env.reset()
            solvable, path = is_map_solvable(grid_map, env.agent_position, env.goal_position)
        
        # print("Map is now solvable!")
        # print(f"Agent: {env.agent_position}, Goal: {env.goal_position}")
        # print(f"Path calculated by A*: {path}")
        
        # print(model.get_env())
        
        model.set_env(env)
        
        # print(model.get_env())
            
        model.learn(total_timesteps=10000, reset_num_timesteps=False)  # Training iteration

        num_evaluation_episodes = 10
        episode_rewards = []

        for i in range(num_evaluation_episodes):
            obs = env.reset()
            solvable, path = is_map_solvable(grid_map, env.agent_position, env.goal_position)
            while not solvable:
                # print("Eval Map not solvable yet!")
                obs = env.reset()
                solvable, path = is_map_solvable(grid_map, env.agent_position, env.goal_position)
            
            # print("Eval Map is now solvable!")
            done = False
            total_reward = 0
            path_taken = []
            # print("Agent init Position: ", env.agent_position)
            # print("Agent's init orientation: ", env.orientation)
            
            print(f"Eval Episode: {i}")
            while not done:
                path_taken.append(env.agent_position)
                
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                if reward == 1000:
                    if len(path) == len(path_taken):
                        print("OPTIMAL PATH TAKEN!!!")
                    # else:
                    #     print(f"Path created by A*: {path}")
                    #     path_taken.append(env.agent_position)
                    print("Goal Reached")
                    # print("Agent final Position: ", env.agent_position)
                    # print("Agent's final orientation: ", env.orientation)
                    #     print("Path taken by agent: ", path_taken)
                total_reward += reward
                # env.render("human")            
            # print(f"Total Reward: {total_reward}")
            episode_rewards.append(total_reward)
            
        avg_reward = np.mean(episode_rewards)
        wandb.log({"reward": avg_reward})  

        print(f"Average reward: {avg_reward}")
        if avg_reward > last_score:
            last_score = avg_reward
            model.save("hybrid_ppo_grid_max3")

        if avg_reward >= desired_avg_reward:
            print("Training completed. Desired average reward achieved.")
            model.save("hybrid_ppo_grid_best3")
            break
        
    wandb.finish()
    
except KeyboardInterrupt:
    print("Training interrupted.")
    model.save("hybrid_ppo_grid_last3")
    print("Last model saved successfully.")
    wandb.finish()