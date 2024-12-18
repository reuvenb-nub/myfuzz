import random
import re
import os
import numpy as np
import torch
import logging
import json
import time

from ..execution import Execution, Tx
from ..ethereum import Method
from .random import PolicyRandom
from .F_random import PolicyFRandom
from .reinforcement.policy_reinforcement_drqn import PolicyReinforcementDRQN
from .reinforcement.policy_reinforcement_ppo_d import PolicyReinforcementPPODiscrete
from .reinforcement.policy_reinforcement_ppo_c import PolicyReinforcementPPOContinuous
from .reinforcement.policy_reinforcement_sac import PolicyReinforcementSAC
from .reinforcement.normalization import Normalization, RewardScaling
import matplotlib.pyplot as plt

has_continuous_action_space = True  # continuous action space; else discrete

max_ep_len = 1000                   # max timesteps in one episode
max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)          # save model frequency (in num timesteps)

action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
#####################################################

## Note : print/log frequencies should be > than max_ep_len

################ PPO hyperparameters ################
update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 80               # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)
#####################################################

def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


LOG = logging.getLogger(__name__)

# max_episode = 100
# start_train = 200

class Environment:

    def __init__(self, limit, seed, max_episode, start_time):
        self.limit = limit
        self.seed = seed
        self.max_episode = max_episode
        self.start_train = 5 * max_episode
        self.start_time = start_time
        
    def MADFuzz_drqn(self, policy, obs, start_time, args):
        if len(policy.contract_manager.fuzz_contract_names) != 1:
            print('please input only one contract to fuzz')
            return
        bug_rate = args.bug_rate

        result = dict()
        result['max_episode'] = self.max_episode
        count_dict = dict()
        count_dict['action'] = dict()
        count_dict['method'] = dict()
        print('fuzz_loop_RL')
        obs.init()

        LOG.info(obs.stat)
        LOG.info('initial calls start')
        self.init_txs(policy, obs, result)
        LOG.info('initial calls end')

        init_limit = 0
        LOG.info('start reinforcement policy')
        result['txs_loop'] = []
        result['bug_finder'] = dict()

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        state, x_method, contract = policy.compute_state(obs)
        hidden = None
        init_episole = 0.7

        if args.mode == 'test':
            episole = 0.15
        else:
            episole = init_episole
        
        hiddens = [None, None, None, None, None, None]

        episode_reward_f = 0
        episode_reward_a = 0
        reward_f_values = []
        reward_a_values = []
        iterations = []

        start_time = time.time()
        i = 1
        elapsed_time = 0
        

        while elapsed_time < self.limit:
            print("step: ", i)
            if i % 1000 == 0:
                for contract_name in policy.contract_manager.fuzz_contract_names:
                    contract = policy.contract_manager[contract_name]
                    policy.execution.set_balance(contract.addresses[0], 10 ** 29)

            if i % self.max_episode < self.max_episode/10:
                tx, action, new_hiddens, arg_actions = policy.select_tx(state, x_method, contract, obs, hiddens=hiddens, frandom=False, episole=1)
            else:
                tx, action, new_hiddens, arg_actions = policy.select_tx(state, x_method, contract, obs, hiddens=hiddens, frandom=False, episole=episole)
                
            for j, new_hidden in enumerate(new_hiddens):
                if new_hidden is not None:
                    hiddens[j] = new_hidden

            if tx is None:
                break
            next_state, new_cov_reward, done, x_method, contract = policy.step(tx, obs)

            reward_f = 0

            action_cov = np.zeros(policy.action_size)
            method_cov = obs.record_manager.get_method_coverage(tx.contract)
            valid_action = policy.valid_action[tx.contract]
            for j, action in enumerate(valid_action):
                if len(valid_action[action]) > 0:
                    for method in valid_action[action]:
                        action_cov[j] += method_cov[method]['block_cov']/len(valid_action[action])
                else:
                    action_cov[j] = 1

            action_cov = action_cov.mean()
            new_insn_coverage, new_block_coverage = obs.stat.get_coverage(tx.contract)
            reward_of_bug = 0
            for bug in obs.stat.update_bug:
                if bug in ['Suicidal', 'Leaking', 'Reentrancy']:
                    reward_of_bug = 1

            if new_cov_reward == 0: 
                new_cov_reward = -1
            
            reward_a = 0.7 * reward_of_bug + (0.3) * new_cov_reward

            if i % self.max_episode == 0:
                reward_f = bug_rate * reward_of_bug + (1-bug_rate) * action_cov
                obs.stat.update_bug = dict()

            policy.agent.store_transition(state, action, reward_f,i)
            if len(arg_actions[0]) > 0:
                policy.int_agent.store_transition(state, arg_actions[0][0], reward_a, i)
            if len(arg_actions[1]) > 0:
                policy.uint_agent.store_transition(state, arg_actions[1][0], reward_a, i)
            if len(arg_actions[2]) > 0:
                policy.bool_agent.store_transition(state, arg_actions[2][0], reward_a, i)
            if len(arg_actions[3]) > 0:
                policy.addr_agent.store_transition(state, arg_actions[3][0], reward_a, i)
            if len(arg_actions[4]) > 0:
                policy.byte_agent.store_transition(state, arg_actions[4][0], reward_a, i)
            state = next_state
            episode_reward_f += reward_f
            episode_reward_a += reward_a


            # print(f'iteration: {i}, reward_f: {reward_f}, reward_a: {reward_a}')
            # LOG.info(obs.stat)

            if i % 100 == 0:
                for bug in obs.stat.to_json()[args.contract]['bugs']:
                    if bug not in result['bug_finder']:
                        result['bug_finder'][bug] = dict()
                    for func in obs.stat.to_json()[args.contract]['bugs'][bug]:
                        if func not in result['bug_finder'][bug]:
                            result['bug_finder'][bug][func] = time.time()

            if i % self.max_episode == 0 and i <= 2000:
                result['txs_loop'].append((time.time(),obs.stat.to_json()))

            if i % self.max_episode == 0 and elapsed_time < self.limit:
                reward_f_values.append(episode_reward_f)
                reward_a_values.append(episode_reward_a)
                iterations.append(i/self.max_episode)

                policy.reset()
                policy.reset_dqn_state()
                episode_reward_f = 0
                episode_reward_a = 0
                hiddens = [None, None, None, None, None, None]
                obs.stat.reset_coverage()
                if args.mode == 'train':
                    policy.agent.buffer.create_new_epi()
                    policy.int_agent.buffer.create_new_epi()
                    policy.uint_agent.buffer.create_new_epi()
                    policy.bool_agent.buffer.create_new_epi()
                    policy.addr_agent.buffer.create_new_epi()
                    policy.byte_agent.buffer.create_new_epi()
                if i >= self.start_train:
                    if args.mode == 'train':
                        policy.agent.learn()
                        policy.int_agent.learn()
                        policy.uint_agent.learn()
                        policy.bool_agent.learn()
                        policy.addr_agent.learn()
                        # policy.byte_agent.learn()
                        episole = init_episole - 0.6 * (i - self.start_train)/(self.limit - self.start_train)
            i += 1
            elapsed_time = time.time() - start_time
            print(f'elapsed_time: {elapsed_time}')

            if time.time() - start_time >= args.limit_time:
                break

        plt.figure(figsize=(12, 6))
        plt.plot(iterations, reward_f_values, label='Reward F')
        plt.plot(iterations, reward_a_values, label='Reward A')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Reward Progression')
        plt.legend()
        plt.grid(True)
        os.makedirs(f'result/{args.contract}', exist_ok=True)
        plt.savefig(f'result/{args.contract}/drqn_reward_plot.png')
        plt.close()
        
        if args.mode == 'train':
            policy.agent.save(args.rl_model)
            # policy.int_agent.save(args.rl_model)
            # policy.uint_agent.save(args.rl_model)
            # policy.bool_agent.save(args.rl_model)
            # policy.addr_agent.save(args.rl_model)
            # policy.byte_agent.save(args.rl_model)
        # print(f'total rewoard:{total_reward}')
        # print(policy.action_trace)
        # print(policy.action_count_array)
        LOG.info(obs.stat)
        with open(f'result/{args.contract}/drqn_result.json', 'w') as f:
            f.write(json.dumps(obs.stat.to_json()))
        
        # print(policy.epi_iter)
        return result
    

    def MADFuzz_ppo_discrete(self, policy, obs, start_time, args):
        if len(policy.contract_manager.fuzz_contract_names) != 1:
            print('please input only one contract to fuzz')
            return
        bug_rate = args.bug_rate

        result = dict()
        result['max_episode'] = self.max_episode
        count_dict = dict()
        count_dict['action'] = dict()
        count_dict['method'] = dict()
        print('fuzz_loop_RL')
        obs.init()

        LOG.info(obs.stat)
        LOG.info('initial calls start')
        self.init_txs(policy, obs, result)
        LOG.info('initial calls end')

        init_limit = 0
        LOG.info('start reinforcement policy')
        result['txs_loop'] = []
        result['bug_finder'] = dict()

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        state, x_method, contract = policy.compute_state(obs)
        hidden = None
        init_episole = 0.7

        if args.mode == 'test':
            episole = 0.15
        else:
            episole = init_episole
        
        hiddens = [None, None, None, None, None, None]

        episode_reward_f = 0
        episode_reward_a = 0
        reward_f_values = []
        reward_a_values = []
        iterations = []
        start_time = time.time()
        i = 1
        elapsed_time = 0

        while elapsed_time < self.limit:
            print("step: ", i)
            if i % 1000 == 0:
                for contract_name in policy.contract_manager.fuzz_contract_names:
                    contract = policy.contract_manager[contract_name]
                    policy.execution.set_balance(contract.addresses[0], 10 ** 29)

            if i % self.max_episode < self.max_episode/10:
                tx, action, new_hiddens, arg_actions = policy.select_tx(state, x_method, contract, obs, hiddens=hiddens, frandom=False, episole=1)
            else:
                tx, action, new_hiddens, arg_actions = policy.select_tx(state, x_method, contract, obs, hiddens=hiddens, frandom=False, episole=episole)
                
            for j, new_hidden in enumerate(new_hiddens):
                if new_hidden is not None:
                    hiddens[j] = new_hidden

            if tx is None:
                break
            next_state, new_cov_reward, done, x_method, contract = policy.step(tx, obs)

            reward_f = 0

            action_cov = np.zeros(policy.action_size)
            method_cov = obs.record_manager.get_method_coverage(tx.contract)
            valid_action = policy.valid_action[tx.contract]
            for j, action in enumerate(valid_action):
                if len(valid_action[action]) > 0:
                    for method in valid_action[action]:
                        action_cov[j] += method_cov[method]['block_cov']/len(valid_action[action])
                else:
                    action_cov[j] = 1

            action_cov = action_cov.mean()
            new_insn_coverage, new_block_coverage = obs.stat.get_coverage(tx.contract)
            reward_of_bug = 0
            for bug in obs.stat.update_bug:
                if bug in ['Suicidal', 'Leaking', 'Reentrancy']:
                    reward_of_bug = 1

            if new_cov_reward == 0:
                new_cov_reward = -1
            reward_a = 0.7 * reward_of_bug + (0.3) * new_cov_reward

            if i % self.max_episode == 0:
                reward_f = bug_rate * reward_of_bug + (1-bug_rate) * action_cov
                obs.stat.update_bug = dict()

            policy.agent.store_transition(state, action, reward_f,i)
            if len(arg_actions[0]) > 0:
                policy.int_agent.buffer.rewards.append(reward_a)
                policy.int_agent.buffer.is_terminals.append(False)
            if len(arg_actions[1]) > 0:
                policy.uint_agent.buffer.rewards.append(reward_a)
                policy.uint_agent.buffer.is_terminals.append(False)
            if len(arg_actions[2]) > 0:
                policy.bool_agent.store_transition(state, arg_actions[2][0], reward_a, i)
            if len(arg_actions[3]) > 0:
                policy.addr_agent.store_transition(state, arg_actions[3][0], reward_a, i)
            if len(arg_actions[4]) > 0:
                policy.byte_agent.store_transition(state, arg_actions[4][0], reward_a, i)
            state = next_state
            episode_reward_f += reward_f
            episode_reward_a += reward_a

            # LOG.info(obs.stat)

            if i % 100 == 0:
                for bug in obs.stat.to_json()[args.contract]['bugs']:
                    if bug not in result['bug_finder']:
                        result['bug_finder'][bug] = dict()
                    for func in obs.stat.to_json()[args.contract]['bugs'][bug]:
                        if func not in result['bug_finder'][bug]:
                            result['bug_finder'][bug][func] = time.time()

            if i % self.max_episode == 0 and i <= 2000:
                result['txs_loop'].append((time.time(),obs.stat.to_json()))

            if i % self.max_episode == 0 and elapsed_time < self.limit:
                reward_f_values.append(episode_reward_f)
                reward_a_values.append(episode_reward_a)
                iterations.append(i/self.max_episode)    
                policy.reset()
                policy.reset_dqn_state()
                episode_reward_f = 0
                episode_reward_a = 0
                hiddens = [None, None, None, None, None, None]
                obs.stat.reset_coverage()
                if args.mode == 'train':
                    policy.agent.buffer.create_new_epi()
                    policy.bool_agent.buffer.create_new_epi()
                    policy.addr_agent.buffer.create_new_epi()
                    policy.byte_agent.buffer.create_new_epi()
                if i >= self.start_train:
                    if args.mode == 'train':
                        policy.agent.learn()
                        policy.int_agent.update()
                        policy.uint_agent.update()
                        policy.bool_agent.learn()
                        policy.addr_agent.learn()
                        # policy.byte_agent.learn()
                        episole = init_episole - 0.6 * (i - self.start_train)/(self.limit - self.start_train)

            if time.time() - start_time >= args.limit_time:
                break

            i += 1
            elapsed_time = time.time() - start_time
            print(f'elapsed_time: {elapsed_time}')
        
        if args.mode == 'train':
            policy.agent.save(args.rl_model)
            # policy.int_agent.save(args.rl_model)
            # policy.uint_agent.save(args.rl_model)
            # policy.bool_agent.save(args.rl_model)
            # policy.addr_agent.save(args.rl_model)
            # policy.byte_agent.save(args.rl_model)
        
        os.makedirs(f'result/{args.contract}', exist_ok=True)
        LOG.info(obs.stat)
        with open(f'result/{args.contract}/ppo_discrete_result.json', 'w') as f:
            f.write(json.dumps(obs.stat.to_json()))

        plt.figure(figsize=(12, 6))
        plt.plot(iterations, reward_f_values, label='Reward F')
        plt.plot(iterations, reward_a_values, label='Reward A')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Reward Progression')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f'result/{args.contract}/ppo_discrete_reward_plot.png')
        plt.close()
        # print(policy.epi_iter)
        return result
    
    def MADFuzz_ppo_continuous(self, policy, obs, start_time, args):
        if len(policy.contract_manager.fuzz_contract_names) != 1:
            print('please input only one contract to fuzz')
            return
        bug_rate = args.bug_rate

        result = dict()
        result['max_episode'] = self.max_episode
        count_dict = dict()
        count_dict['action'] = dict()
        count_dict['method'] = dict()
        print('fuzz_loop_RL')
        obs.init()

        LOG.info(obs.stat)
        LOG.info('initial calls start')
        self.init_txs(policy, obs, result)
        LOG.info('initial calls end')

        init_limit = 0
        LOG.info('start reinforcement policy')
        result['txs_loop'] = []
        result['bug_finder'] = dict()

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        episode_reward = 0

        state, x_method, contract = policy.compute_state(obs)
        hidden = None
        init_episole = 0.7

        if args.mode == 'test':
            episole = 0.15
        else:
            episole = init_episole
        
        hiddens = [None, None, None, None, None, None]

        episode_reward_f = 0
        episode_reward_a = 0
        reward_f_values = []
        reward_a_values = []
        iterations = []
        start_time = time.time()
        i = 1
        elapsed_time = 0

        while elapsed_time < self.limit:
            print("step: ", i)
            if i % 1000 == 0:
                for contract_name in policy.contract_manager.fuzz_contract_names:
                    contract = policy.contract_manager[contract_name]
                    policy.execution.set_balance(contract.addresses[0], 10 ** 29)

            if i % self.max_episode < self.max_episode/10:
                tx, action, new_hiddens, arg_actions = policy.select_tx(state, x_method, contract, obs, hiddens=hiddens, frandom=False, episole=1)
            else:
                tx, action, new_hiddens, arg_actions = policy.select_tx(state, x_method, contract, obs, hiddens=hiddens, frandom=False, episole=episole)
                
            for j, new_hidden in enumerate(new_hiddens):
                if new_hidden is not None:
                    hiddens[j] = new_hidden

            if tx is None:
                break
            next_state, new_cov_reward, done, x_method, contract = policy.step(tx, obs)

            reward_f = 0

            action_cov = np.zeros(policy.action_size)
            method_cov = obs.record_manager.get_method_coverage(tx.contract)
            valid_action = policy.valid_action[tx.contract]
            for j, action in enumerate(valid_action):
                if len(valid_action[action]) > 0:
                    for method in valid_action[action]:
                        action_cov[j] += method_cov[method]['block_cov']/len(valid_action[action])
                else:
                    action_cov[j] = 1

            action_cov = action_cov.mean()
            new_insn_coverage, new_block_coverage = obs.stat.get_coverage(tx.contract)
            reward_of_bug = 0
            for bug in obs.stat.update_bug:
                if bug in ['Suicidal', 'Leaking', 'Reentrancy']:
                    reward_of_bug = 1
            if new_cov_reward == 0:
                new_cov_reward = -1
            reward_a = 0.7 * reward_of_bug + (0.3) * new_cov_reward

            if i % self.max_episode == 0:
                reward_f = bug_rate * reward_of_bug + (1-bug_rate) * action_cov
                obs.stat.update_bug = dict()

            policy.agent.store_transition(state, action, reward_f,i)
            if len(arg_actions[0]) > 0:
                policy.int_agent.buffer.rewards.append(reward_a)
                policy.int_agent.buffer.is_terminals.append(False)
            if len(arg_actions[1]) > 0:
                policy.uint_agent.buffer.rewards.append(reward_a)
                policy.uint_agent.buffer.is_terminals.append(False)
            if len(arg_actions[2]) > 0:
                policy.bool_agent.store_transition(state, arg_actions[2][0], reward_a, i)
            if len(arg_actions[3]) > 0:
                policy.addr_agent.store_transition(state, arg_actions[3][0], reward_a, i)
            if len(arg_actions[4]) > 0:
                policy.byte_agent.store_transition(state, arg_actions[4][0], reward_a, i)
            state = next_state
            episode_reward_f += reward_f
            episode_reward_a += reward_a
            # LOG.info(obs.stat)

            if i % 100 == 0:
                for bug in obs.stat.to_json()[args.contract]['bugs']:
                    if bug not in result['bug_finder']:
                        result['bug_finder'][bug] = dict()
                    for func in obs.stat.to_json()[args.contract]['bugs'][bug]:
                        if func not in result['bug_finder'][bug]:
                            result['bug_finder'][bug][func] = time.time()

            if i % self.max_episode == 0 and i <= 2000:
                result['txs_loop'].append((time.time(),obs.stat.to_json()))

            if i % self.max_episode == 0 and elapsed_time < self.limit:
                reward_f_values.append(episode_reward_f)
                reward_a_values.append(episode_reward_a)
                iterations.append(i/self.max_episode) 
                policy.reset()
                policy.reset_dqn_state()
                episode_reward_f = 0
                episode_reward_a = 0
                hiddens = [None, None, None, None, None, None]
                obs.stat.reset_coverage()
                if args.mode == 'train':
                    policy.agent.buffer.create_new_epi()
                    policy.bool_agent.buffer.create_new_epi()
                    policy.addr_agent.buffer.create_new_epi()
                    policy.byte_agent.buffer.create_new_epi()
                if i >= self.start_train:
                    if args.mode == 'train':
                        policy.agent.learn()
                        policy.int_agent.update()
                        policy.uint_agent.update()
                        policy.bool_agent.learn()
                        policy.addr_agent.learn()
                        # policy.byte_agent.learn()
                        episole = init_episole - 0.6 * (i - self.start_train)/(self.limit - self.start_train)

            i += 1
            elapsed_time = time.time() - start_time
            print(f'elapsed_time: {elapsed_time}')

            if time.time() - start_time >= args.limit_time:
                break
        
        if args.mode == 'train':
            policy.agent.save(args.rl_model)
            # policy.int_agent.save(args.rl_model)
            # policy.uint_agent.save(args.rl_model)
            # policy.bool_agent.save(args.rl_model)
            # policy.addr_agent.save(args.rl_model)
            # policy.byte_agent.save(args.rl_model)
        # print(f'total rewoard:{total_reward}')
        # print(policy.action_trace)
        # print(policy.action_count_array)
        os.makedirs(f'result/{args.contract}', exist_ok=True)
        LOG.info(obs.stat)
        with open(f'result/{args.contract}/ppo_continuous_result.json', 'w') as f:
            f.write(json.dumps(obs.stat.to_json()))

        plt.figure(figsize=(12, 6))
        plt.plot(iterations, reward_f_values, label='Reward F')
        plt.plot(iterations, reward_a_values, label='Reward A')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Reward Progression')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f'result/{args.contract}/ppo_continuous_reward_plot.png')
        plt.close()

        # print(policy.epi_iter)
        return result

    def MADFuzz_sac(self, policy, obs, start_time, args):
        if len(policy.contract_manager.fuzz_contract_names) != 1:
            print('please input only one contract to fuzz')
            return
        bug_rate = args.bug_rate

        result = dict()
        result['max_episode'] = self.max_episode
        count_dict = dict()
        count_dict['action'] = dict()
        count_dict['method'] = dict()
        print('fuzz_loop_RL')
        obs.init()

        LOG.info(obs.stat)
        LOG.info('initial calls start')
        self.init_txs(policy, obs, result)
        LOG.info('initial calls end')

        init_limit = 0
        LOG.info('start reinforcement policy')
        result['txs_loop'] = []
        result['bug_finder'] = dict()

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        episode_reward = 0

        state, x_method, contract = policy.compute_state(obs)
        hidden = None
        init_episole = 0.7

        if args.mode == 'test':
            episole = 0.15
        else:
            episole = init_episole
        
        hiddens = [None, None, None, None, None, None]

        episode_reward_f = 0
        episode_reward_a = 0
        reward_f_values = []
        reward_a_values = []
        iterations = []
        start_time = time.time()
        i = 1
        elapsed_time = 0

        while elapsed_time < self.limit:
            print("step: ", i)
            if i % 1000 == 0:
                for contract_name in policy.contract_manager.fuzz_contract_names:
                    contract = policy.contract_manager[contract_name]
                    policy.execution.set_balance(contract.addresses[0], 10 ** 29)

            if i % self.max_episode < self.max_episode/10:
                tx, action, new_hiddens, arg_actions = policy.select_tx(state, x_method, contract, obs, hiddens=hiddens, frandom=False, episole=1)
            else:
                tx, action, new_hiddens, arg_actions = policy.select_tx(state, x_method, contract, obs, hiddens=hiddens, frandom=False, episole=episole)
                
            for j, new_hidden in enumerate(new_hiddens):
                if new_hidden is not None:
                    hiddens[j] = new_hidden

            if tx is None:
                break
            next_state, new_cov_reward, done, x_method, contract = policy.step(tx, obs)

            reward_f = 0

            action_cov = np.zeros(policy.action_size)
            method_cov = obs.record_manager.get_method_coverage(tx.contract)
            valid_action = policy.valid_action[tx.contract]
            for j, action in enumerate(valid_action):
                if len(valid_action[action]) > 0:
                    for method in valid_action[action]:
                        action_cov[j] += method_cov[method]['block_cov']/len(valid_action[action])
                else:
                    action_cov[j] = 1

            action_cov = action_cov.mean()
            new_insn_coverage, new_block_coverage = obs.stat.get_coverage(tx.contract)
            reward_of_bug = 0
            for bug in obs.stat.update_bug:
                if bug in ['Suicidal', 'Leaking', 'Reentrancy']:
                    reward_of_bug = 1

            if new_cov_reward == 0:
                new_cov_reward = -1
            reward_a = 0.7 * reward_of_bug + (0.3) * new_cov_reward

            if i % self.max_episode == 0:
                reward_f = bug_rate * reward_of_bug + (1-bug_rate) * action_cov
                obs.stat.update_bug = dict()

            policy.agent.store_transition(state, action, reward_f,i)
            if len(arg_actions[0]) > 0:
                policy.int_agent.store(state, reward_a, False, action, next_state)
            if len(arg_actions[1]) > 0:
                policy.uint_agent.store(state, reward_a, False, action, next_state)
            if len(arg_actions[2]) > 0:
                policy.bool_agent.store_transition(state, arg_actions[2][0], reward_a, i)
            if len(arg_actions[3]) > 0:
                policy.addr_agent.store_transition(state, arg_actions[3][0], reward_a, i)
            if len(arg_actions[4]) > 0:
                policy.byte_agent.store_transition(state, arg_actions[4][0], reward_a, i)
            state = next_state
            episode_reward_f += reward_f
            episode_reward_a += reward_a
            # LOG.info(obs.stat)

            if i % 100 == 0:
                for bug in obs.stat.to_json()[args.contract]['bugs']:
                    if bug not in result['bug_finder']:
                        result['bug_finder'][bug] = dict()
                    for func in obs.stat.to_json()[args.contract]['bugs'][bug]:
                        if func not in result['bug_finder'][bug]:
                            result['bug_finder'][bug][func] = time.time()

            if i % self.max_episode == 0 and i <= 2000:
                result['txs_loop'].append((time.time(),obs.stat.to_json()))

            if i % self.max_episode == 0 and elapsed_time < self.limit:
                reward_f_values.append(episode_reward_f)
                reward_a_values.append(episode_reward_a)
                iterations.append(i/self.max_episode) 
                policy.reset()
                policy.reset_dqn_state()
                episode_reward_f = 0
                episode_reward_a = 0
                hiddens = [None, None, None, None, None, None]
                obs.stat.reset_coverage()
                if args.mode == 'train':
                    policy.agent.buffer.create_new_epi()
                    policy.bool_agent.buffer.create_new_epi()
                    policy.addr_agent.buffer.create_new_epi()
                    policy.byte_agent.buffer.create_new_epi()
                if i >= self.start_train:
                    if args.mode == 'train':
                        policy.agent.learn()
                        policy.int_agent.train()
                        policy.uint_agent.train()
                        policy.bool_agent.learn()
                        policy.addr_agent.learn()
                        # policy.byte_agent.learn()
                        episole = init_episole - 0.6 * (i - self.start_train)/(self.limit - self.start_train)

            if time.time() - start_time >= args.limit_time:
                break

            i += 1
            elapsed_time = time.time() - start_time
            print(f'elapsed_time: {elapsed_time}')
        
        if args.mode == 'train':
            policy.agent.save(args.rl_model)
            # policy.int_agent.save(args.rl_model)
            # policy.uint_agent.save(args.rl_model)
            # policy.bool_agent.save(args.rl_model)
            # policy.addr_agent.save(args.rl_model)
            # policy.byte_agent.save(args.rl_model)
        # print(f'total rewoard:{total_reward}')
        # print(policy.action_trace)
        # print(policy.action_count_array)
        LOG.info(obs.stat)
        os.makedirs(f'result/{args.contract}', exist_ok=True)
        with open(f'result/{args.contract}/sac_result.json', 'w') as f:
            f.write(json.dumps(obs.stat.to_json()))

        plt.figure(figsize=(12, 6))
        plt.plot(iterations, reward_f_values, label='Reward F')
        plt.plot(iterations, reward_a_values, label='Reward A')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Reward Progression')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f'result/{args.contract}/sac_reward_plot.png')
        plt.close()
        # print(policy.epi_iter)
        return result

    def fuzz_loop_RL(self, policy, obs, start_time, args):
        if len(policy.contract_manager.fuzz_contract_names) != 1:
            print('please input only one contract to fuzz')
            return
        bug_rate = args.bug_rate

        result = dict()
        result['max_episode'] = self.max_episode
        count_dict = dict()
        count_dict['action'] = dict()
        count_dict['method'] = dict()
        print('fuzz_loop_RL')
        obs.init()

        LOG.info(obs.stat)
        LOG.info('initial calls start')
        self.init_txs(policy, obs, result)
        LOG.info('initial calls end')

        init_limit = 0
        LOG.info('start reinforcement policy')
        result['txs_loop'] = []
        result['bug_finder'] = dict()

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # without rnn
        episode_reward = 0

        # hidden = policy.ddpg.get_initial_states()
        state, x_method, contract = policy.compute_state(obs)
        hidden = None
        init_episole = 0.7

        if args.mode == 'test':
            episole = 0.15
        else:
            episole = init_episole

        for i in range(1+init_limit, self.limit+1):
            if i % 1000 == 0:
                # reset the balance of the contracts
                for contract_name in policy.contract_manager.fuzz_contract_names:
                    contract = policy.contract_manager[contract_name]
                    policy.execution.set_balance(contract.addresses[0], 10 ** 29)
            # print(state)
            # state = np.random.random(115)
            if i % self.max_episode < self.max_episode/10:
                tx, action, hidden, arg_actions = policy.select_tx(state, x_method, contract, obs, hidden=hidden, frandom=False, episole=1)
            else:
                tx, action, hidden, arg_actions = policy.select_tx(state, x_method, contract, obs, hidden=hidden, frandom=False, episole=episole)
            
            if tx is None:
                break
            next_state, reward, done, x_method, contract = policy.step(tx, obs)
            exit(1)
            
            # print(reward)
            
            # print('state: ', np.linalg.norm(next_state-state))
            # input('stop')

            # print(state, action, reward, done)
            # policy.agent.store_transition(state, action, reward, next_state, done, contract)
            reward = 0
            if i % self.max_episode == 0:
                action_cov = np.zeros(policy.action_size)
                method_cov = obs.record_manager.get_method_coverage(tx.contract)
                # print(method_cov)
                valid_action = policy.valid_action[tx.contract]
                # print(valid_action)
                for j, action in enumerate(valid_action):
                    # print(j, action)
                    # print(valid_action[action])
                    if len(valid_action[action]) > 0:
                        for method in valid_action[action]:
                            action_cov[j] += method_cov[method]['block_cov']/len(valid_action[action])
                    else:
                        action_cov[j] = 1
                # print('action_cov: ', action_cov)
                # input('stop')
                action_cov = action_cov.mean()
                # print(action_cov)
                new_insn_coverage, new_block_coverage = obs.stat.get_coverage(tx.contract)
                # print(new_insn_coverage, new_block_coverage)
                # exit(1)
                reward_of_bug = 0
                for bug in obs.stat.update_bug:
                    if bug in ['Suicidal', 'Leaking', 'Reentrancy', 'UnhandledException']:
                        # reward_of_bug += len(obs.stat.update_bug[bug])
                        reward_of_bug = 1
                reward = bug_rate * reward_of_bug + (1-bug_rate) * action_cov
                # print('reward: ',reward)
                obs.stat.update_bug = dict()

            policy.agent.store_transition(state, action, reward,i)
            if len(arg_actions[0]) > 0:
                policy.int_agent.store_transition(state, arg_actions[0][0], reward,i)
            if len(arg_actions[0]) > 0:
                policy.uint_agent.store_transition(state, arg_actions[1][0], reward,i)
            if len(arg_actions[0]) > 0:
                policy.bool_agent.store_transition(state, arg_actions[2][0], reward,i)
            if len(arg_actions[0]) > 0:
                policy.addr_agent.store_transition(state, arg_actions[3][0], reward,i)
            if len(arg_actions[0]) > 0:
                policy.byte_agent.store_transition(state, arg_actions[4][0], reward,i)
            
            state = next_state
            episode_reward += reward
            # LOG.info(obs.stat)

            if i % 100 == 0:
                for bug in obs.stat.to_json()[args.contract]['bugs']:
                    if bug not in result['bug_finder']:
                        result['bug_finder'][bug] = dict()
                    for func in obs.stat.to_json()[args.contract]['bugs'][bug]:
                        if func not in result['bug_finder'][bug]:
                            result['bug_finder'][bug][func] = time.time()

            if i % self.max_episode == 0 and i <= 2000:
                result['txs_loop'].append((time.time(),obs.stat.to_json()))
                # print(policy.action_trace)
                # print(policy.action_count_array)
                # print(policy.agent_action_count_array)
            if i % self.max_episode == 0 and i < self.limit:
                policy.reset()
                policy.reset_dqn_state()
                episode_reward = 0
                hidden = None
                obs.stat.reset_coverage()
                if args.mode == 'train':
                    policy.agent.buffer.create_new_epi()
                    # policy.int_agent.buffer.create_new_epi()
                    policy.uint_agent.buffer.create_new_epi()
                    # policy.bool_agent.buffer.create_new_epi()
                    # policy.addr_agent.buffer.create_new_epi()
                    # policy.byte_agent.buffer.create_new_epi()
                if i >= self.start_train:
                    # policy.agent.buffer.print_info()
                    if args.mode == 'train':
                        policy.agent.learn()
                        policy.int_agent.learn()
                        policy.uint_agent.learn(test=True)
                        policy.bool_agent.learn()
                        policy.addr_agent.learn()
                        # policy.byte_agent.learn()
                        # exit(1)
                        episole = init_episole - 0.6 * (i - self.start_train)/(self.limit - self.start_train)
                        # print(episole)
                        # input('stop')
                # if time.time() - self.start_time > (self.limit_time - 1):
                #     break
            if time.time() - start_time >= args.limit_time:
                break

        if args.mode == 'train':
            policy.agent.save(args.rl_model)
        # print(f'total rewoard:{total_reward}')
        # print(policy.action_trace)
        # print(policy.action_count_array)
        LOG.info(obs.stat)
        # print(policy.epi_iter)
        return result, count_dict

    def init_txs_RL(self, policy, obs, limit, result):
        """
        send some tx to call some method in the fuzz_contract, include FALLBACK and the methods that not payable
        """
        policy_frandom = PolicyFRandom(policy.execution, policy.contract_manager, policy.account_manager)
        result['init_txs_RL'] = []
        for _ in range(limit):
            tx = policy_frandom.select_tx(obs)
            logger = policy_frandom.execution.commit_tx(tx)
            obs.update(logger, True)
        LOG.info(obs.stat)
        result['init_txs_RL'].append(obs.stat.to_json())


    def init_txs(self, policy, obs, result):
        """
        send some tx to call some method in the fuzz_contract, include FALLBACK and the methods that not payable
        """
        policy_random = PolicyRandom(policy.execution, policy.contract_manager, policy.account_manager)
        result['init_txs'] = []
        for name in policy.contract_manager.fuzz_contract_names:
            contract = policy.contract_manager[name]
            if Method.FALLBACK not in contract.abi.methods_by_name:
                tx = Tx(policy_random, contract.name, contract.addresses[0], Method.FALLBACK, bytes(), [], 0, 0, 0, True)
                logger = policy_random.execution.commit_tx(tx)
                obs.update(logger, True)
                # LOG.info(obs.stat)
                result['init_txs'].append(obs.stat.to_json())

            for method in contract.abi.methods:
                # print(method.name)
                if not contract.is_payable(method.name):
                    # print(method.name)
                    tx = policy_random.select_tx_for_method(contract, method, obs)
                    tx.amount = 1
                    logger = policy_random.execution.commit_tx(tx)
                    obs.update(logger, True)
                    # LOG.info(obs.stat)
                    result['init_txs'].append(obs.stat.to_json())

    def fuzz_loop(self, policy, obs):
        obs.init()

        result = dict()
        LOG.info(obs.stat)
        LOG.info('initial calls start')
        self.init_txs(policy, obs, result)
        LOG.info('initial calls end')

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        result['txs_loop'] = []
        result['max_episode'] = self.max_episode

        for i in range(1, self.limit+1):
            if policy.__class__ in (PolicyRandom, PolicyFRandom) and i > self.limit // 2:
                # reset the balance of the contracts
                for contract_name in policy.contract_manager.fuzz_contract_names:
                    contract = policy.contract_manager[contract_name]
                    policy.execution.set_balance(contract.addresses[0], 10 ** 29)
            
            tx = policy.select_tx(obs)
            if tx is None:
                break

            logger = policy.execution.commit_tx(tx)
            # print(logger)
            old_insn_coverage = obs.stat.get_insn_coverage(tx.contract)
            obs.update(logger, False)
            new_insn_coverage = obs.stat.get_insn_coverage(tx.contract)

            if i % self.max_episode == 0:
                result['txs_loop'].append(obs.stat.to_json())


            if i % self.max_episode == 0:
                # reset the state
                policy.reset()
        
        LOG.info(obs.stat)
        return result
