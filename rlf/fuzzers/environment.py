import random
import re
import numpy as np
import torch
import logging
import json
import time

from ..execution import Execution, Tx
from ..ethereum import Method
from .random import PolicyRandom
from .F_random import PolicyFRandom
from .reinforcement.policy_reinforcement import PolicyReinforcement


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
        
    def MADFuzz_run(self, policy, obs, start_time, args):
        if len(policy.contract_manager.fuzz_contract_names) != 1:
            print('please input only one contract to fuzz')
            return
        bug_rate = args.bug_rate

        result = dict()
        result['max_episode'] = self.max_episode

        print('MADFuzz_run')
        
        obs.init()

        LOG.info(obs.stat)
        LOG.info('initial calls start')
        self.init_txs(policy, obs, result)
        LOG.info('initial calls end')

        init_limit = 0
        LOG.info('start reinforcement policy')
        result['txs_loop'] = []
        result['bug_finder'] = dict()

        
        init_episole = 0.7

        if args.mode == 'test':
            episole = 0.15
        else:
            episole = init_episole       

        for episode in range(1, self.limit+1):
            LOG.info(f'Episode: {episode}')
            
            epi_obs = obs
            policy.reset()
            policy.reset_dqn_state()
            episode_reward = 0
            
            state, x_method, contract = policy.compute_state(epi_obs)
            hiddens = [None, None, None, None, None, None]

        
            if args.mode == 'train':
                policy.agent.buffer.create_new_epi()
                policy.int_agent.buffer.create_new_epi()
                policy.uint_agent.buffer.create_new_epi()
                policy.bool_agent.buffer.create_new_epi()
                policy.addr_agent.buffer.create_new_epi()
                policy.byte_agent.buffer.create_new_epi()
            

            for step in range(1, self.max_episode+1):
                tx, action, new_hiddens, arg_actions = policy.select_tx(state, x_method, contract, epi_obs, hiddens=hiddens, frandom=False, episole=episole)
                
                for i, new_hidden in enumerate(new_hiddens):
                    if new_hidden is not None:
                        hiddens[i] = new_hidden
                
                if tx is None:
                    break
                next_state, new_cov_reward, done, x_method, contract = policy.step(tx, epi_obs)
                
                action_cov = 0
                if step == self.max_episode:
                    action_cov = np.zeros(policy.action_size)
                    method_cov = epi_obs.record_manager.get_method_coverage(tx.contract)
                    valid_action = policy.valid_action[tx.contract]
                    for j, action in enumerate(valid_action):
                        if len(valid_action[action]) > 0:
                            for method in valid_action[action]:
                                action_cov[j] += method_cov[method]['block_cov']/len(valid_action[action])
                        else:
                            action_cov[j] = 1
                            
                    print("method_cov: ", method_cov)
                    
                    action_cov = action_cov.mean()
                    new_insn_coverage, new_block_coverage = epi_obs.stat.get_coverage(tx.contract)
                    
                reward_of_bug = 0
                for bug in epi_obs.stat.update_bug:
                    if bug in ['Suicidal', 'Leaking', 'Reentrancy']:
                        reward_of_bug = 1
                        
                if new_cov_reward == 0: 
                    new_cov_reward = -1
                    
                reward_f = 0
                if step == self.max_episode:
                    reward_f = 0.7 * reward_of_bug + (0.3) * action_cov
                reward_a = 0.7 * reward_of_bug + (0.3) * new_cov_reward
                
                policy.agent.store_transition(state, action, reward_f, step*episode)
                if len(arg_actions[0]) > 0:
                    policy.int_agent.store_transition(state, arg_actions[0][0], reward_a, step*episode)
                if len(arg_actions[1]) > 0:
                    policy.uint_agent.store_transition(state, arg_actions[1][0], reward_a, step*episode)
                if len(arg_actions[2]) > 0:
                    policy.bool_agent.store_transition(state, arg_actions[2][0], reward_a, step*episode)
                if len(arg_actions[3]) > 0:
                    policy.addr_agent.store_transition(state, arg_actions[3][0], reward_a, step*episode)
                if len(arg_actions[4]) > 0:
                    policy.byte_agent.store_transition(state, arg_actions[4][0], reward_a, step*episode)
                    
                state = next_state
                episode_reward += reward_f
                    
            if args.mode == 'train':
                # policy.agent.buffer.print_info()
                policy.agent.learn()
                policy.int_agent.learn()
                policy.uint_agent.learn()
                policy.bool_agent.learn()
                policy.addr_agent.learn()
                # policy.byte_agent.learn()
                # exit(1)
                episole = init_episole - 0.6 * (step*episode - self.start_train)/(self.limit*50 - self.start_train)
                
                
            for bug in obs.stat.to_json()[args.contract]['bugs']:
                if bug not in result['bug_finder']:
                    result['bug_finder'][bug] = dict()
                for func in obs.stat.to_json()[args.contract]['bugs'][bug]:
                    if func not in result['bug_finder'][bug]:
                        result['bug_finder'][bug][func] = time.time()

                result['txs_loop'].append((time.time(),obs.stat.to_json()))
                    
                if time.time() - start_time >= args.limit_time:
                    break
        
        if args.mode == 'train':
            policy.agent.save(args.rl_model)
        # print(f'total rewoard:{total_reward}')
        # print(policy.action_trace)
        # print(policy.action_count_array)
        LOG.info(obs.stat)
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
