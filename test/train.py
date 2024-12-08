import torch
import torch.optim as optim
import time
from tqdm import tqdm
from environment import BreakoutEnv
from policy import PolicyNetwork
from constants import *
import pygame
from torch.distributions import Normal

def train_agent():
    try:
        print("Initializing training environment...")
        print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        env = BreakoutEnv(render_mode="human")
        state_size = len(env.reset())
        print("State size:", state_size)
        
        policy = PolicyNetwork(state_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
        #optimizer = optim.SGD(policy.parameters(), lr=0.01, momentum=0.9)
        print("Policy network initialized successfully")
        
        # Initialize progress bar
        pbar = tqdm(range(MAX_EPISODES), desc="Training Progress")
        running_reward = 0
        start_time = time.time()
        
        for episode in pbar:
            state = env.reset()  # Reset at start of each episode
            log_probs = []
            rewards = []
            episode_reward = 0
            steps_in_episode = 0
            
            try:
                while steps_in_episode < MAX_STEPS_PER_EPISODE:
                    steps_in_episode += 1
                    
                    # Get action from policy
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(
                        torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    )
                    
                    mean_std = policy(state_tensor)
                    mean, std = mean_std[0, 0], torch.exp(mean_std[0, 1])
                    dist = Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    # Clip action to valid range [-1, 1]
                    #
                    action = torch.clamp(action, -1, 1)
                    
                    # Step environment
                    try:
                        # Add additional random noise to action for exploration
                        #noise = torch.randn(1) * 0.3  # Add explicit exploration noise
                        #action = action + noise
                        action = torch.clamp(action, -1, 1)
                        
                        next_state, reward, done = env.step(action.item())
                        
                        # Give small reward for moving toward the ball
                        ball_x = next_state[1] * SCREEN_WIDTH  # Denormalize ball x position
                        paddle_x = next_state[0] * SCREEN_WIDTH  # Denormalize paddle x position
                        movement_reward = -abs((ball_x - (paddle_x + PADDLE_WIDTH/2)) / SCREEN_WIDTH)
                        reward += movement_reward * 5  # Scale the movement reward
                        if done and reward > 0:  # Assuming 'done' indicates a successful hit
                            reward += 10  # Add a larger reward for hitting the ball
                        
                        log_probs.append(log_prob)
                        rewards.append(reward)
                        episode_reward += reward
                        
                        # Handle game over
                        if done:
                            print(f"\nGame Over! Score: {env.score} Steps: {steps_in_episode}")
                            # Don't break, just reset and continue
                            state = env.reset()
                        else:
                            state = next_state
                            
                        # Handle Pygame events
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                print("\nClosing game window...")
                                pygame.quit()
                                return
                            
                    except Exception as e:
                        print(f"\nError during environment step: {e}")
                        state = env.reset()  # Reset on error
                        continue
                
                # Update policy only if we have enough experiences
                if len(rewards) > 0:  # Only update if we have collected some experiences
                    try:
                        R = 0
                        policy_loss = []
                        returns = []
                        
                        for r in rewards[::-1]:
                            R = r + GAMMA * R
                            returns.insert(0, R)
                        
                        if len(returns) > 0:  # Double check we have returns
                            returns = torch.FloatTensor(returns).to(
                                torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            )
                            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                            
                            for log_prob, R in zip(log_probs, returns):
                                policy_loss.append(-log_prob * R)
                            
                            if len(policy_loss) > 0:  # Make sure we have losses to optimize
                                optimizer.zero_grad()
                                policy_loss = torch.cat(policy_loss)  # Concatenate losses
                                if policy_loss.numel() > 0:  # Check if tensor is not empty
                                    loss = policy_loss.sum()
                                    loss.backward()
                                    optimizer.step()
                                    print(f"\nUpdated policy - Loss: {loss.item():.3f}")
                                else:
                                    print("\nWarning: Empty policy loss tensor")
                            else:
                                print("\nWarning: No policy losses collected")
                        else:
                            print("\nWarning: No returns calculated")
                    except Exception as e:
                        print(f"\nError during policy update: {e}")
                        print(f"Rewards collected: {len(rewards)} (first 5: {rewards[:5]})")
                        print(f"Log probs collected: {len(log_probs)} (first 5: {log_probs[:5]})")
                        continue  # Skip this update but continue training
                else:
                    print("\nWarning: No experiences collected in this episode")
                    
                    # Update running reward and progress bar
                    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
                    
                    elapsed_time = time.time() - start_time
                    avg_time_per_episode = elapsed_time / (episode + 1)
                    remaining_episodes = MAX_EPISODES - (episode + 1)
                    estimated_remaining_time = remaining_episodes * avg_time_per_episode
                    
                    pbar.set_description(
                        f"Episode {episode + 1}/{MAX_EPISODES} | "
                        f"Reward: {episode_reward:.1f} | "
                        f"Running Reward: {running_reward:.1f} | "
                        f"Steps: {steps_in_episode}"
                    )
                
            except Exception as e:
                print(f"\nError during episode: {e}")
                continue  # Continue with next episode on error
            
            # Save checkpoint every 50 episodes
            if (episode + 1) % 50 == 0:
                try:
                    torch.save({
                        'episode': episode,
                        'model_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'running_reward': running_reward
                    }, f'breakout_pg_checkpoint_{episode + 1}.pt')
                except Exception as e:
                    print(f"Warning: Could not save checkpoint: {e}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
        torch.save({
            'episode': episode,
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'running_reward': running_reward
        }, 'breakout_pg_interrupted.pt')
    except Exception as e:
        print(f"\nError during training: {e}")
        raise e
    finally:
        print("\nTraining completed or interrupted. Final statistics:")
        print(f"Total episodes completed: {episode + 1}")
        print(f"Final running reward: {running_reward:.1f}")
        print(f"Total training time: {(time.time() - start_time)/60:.1f} minutes")