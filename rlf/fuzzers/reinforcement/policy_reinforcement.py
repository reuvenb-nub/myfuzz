import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.externals import joblib
import os
import math

from .DRQN import DRQN, device, use_cuda

from ..random import PolicyRandom
from ..policy_base import PolicyBase
from ...ethereum import SolType
from ...ethereum.evm.opcode import *
from ...execution import Tx
import numpy as np

from ..imitation.models import PolicyNet
from ..imitation.nlp import NLP
from ..seed.int_values import INT_VALUES_FREQUENT, INT_VALUES_UNFREQUENT
from ..seed.amounts import AMOUNTS
from ..seed.addr_map import ADDR_MAP
ADDR_FEAT = 10
RNN_HIDDEN_SIZE = 100
NUM_LAYERS = 1
RAW_FEATURE_SIZE = 65 + 300
INT_EXPLORE_RATE = -1
ACTION_SIZE = 5
BOOL_SIZE = 2
BYTE_SIZE = 256

def get_decay(epi_iter):
    decay = math.pow(0.95, epi_iter)
    if decay < 0.2:
        decay = 0.2
    return decay

classification_list = ['pay-call','nopay-call','pay-nocall','nopay-nocall-store','selfdestruct']

class PolicyReinforcement(PolicyBase):

    def __init__(self, execution, contract_manager, account_manager, args):
        super().__init__(execution, contract_manager, account_manager)
        self.args = args

        self.addr_map = ADDR_MAP
        self.int_values_frequent = INT_VALUES_FREQUENT
        self.int_values_unfrequent = INT_VALUES_UNFREQUENT
        self.amounts = AMOUNTS
        self.slice_size = 2

        self.raw_feature_size = RAW_FEATURE_SIZE
        self.feature_size = RNN_HIDDEN_SIZE
        self.state_size = RNN_HIDDEN_SIZE

        self.action_size = ACTION_SIZE

        self.agent = DRQN(state_dim=110+ACTION_SIZE, action_dim=self.action_size)
        
        self.int_agent = DRQN(state_dim=110+ACTION_SIZE, action_dim=255)
        self.uint_agent = DRQN(state_dim=110+ACTION_SIZE, action_dim=257)
        self.bool_agent = DRQN(state_dim=110+ACTION_SIZE, action_dim=BOOL_SIZE)
        self.addr_agent = DRQN(state_dim=110+ACTION_SIZE, action_dim=len(self.addresses))
        self.byte_agent = DRQN(state_dim=110+ACTION_SIZE, action_dim=BYTE_SIZE)

        # dqn state
        self.tx_count_dqn = 0
        self.action_count_array = np.zeros(self.action_size)
        self.init_action_space()
        self.max_episode = args.max_episode
        self.trace_bow_accumulative = select_trace_opcode([0 for _ in range(256)])
        self.action_trace = list()
        self.epi_iter = 0
        self.agent_action_count_array = np.zeros(self.action_size)
        
        self.int_agent_action_count_array = np.zeros(255)
        self.uint_agent_action_count_array = np.zeros(257)
        self.bool_agent_action_count_array = np.zeros(2)
        self.addr_agent_action_count_array = np.zeros(len(self.addresses))
        self.byte_agent_action_count_array = np.zeros(256)

        # self.graphs_col = GraphsCollection()
        self.last_method = dict()
        self.method_names = {}
        self.method_bows = {}
        self.nlp = NLP()
        self.nlp.w2v = pickle.load(open('ilf_w2v.pkl', 'rb'))

        self.compress_net = PolicyNet(self.raw_feature_size, self.feature_size, self.state_size).to(device)
        self.scaler = None
        self.mode = args.mode
        
        self.int_actions = []
        self.uint_actions = []
        self.bool_actions = []
        self.addr_actions = []
        self.byte_actions = []

    def init_action_space(self):
        # classify the function
        self.valid_action = dict()
        self.method_insn_length = dict()

        for contract_name in self.contract_manager.fuzz_contract_names:
            self.valid_action[contract_name] = dict()
            contract = self.contract_manager[contract_name]
            abi_json = contract.abi.to_json()
            self.valid_action[contract_name], self.method_name_to_index, self.index_to_method_name = self.classification_by_pattern(abi_json)
            for index, name in enumerate(abi_json['methods'].keys()):
                method = abi_json['methods'][name]
                self.method_insn_length[name] = len(method['insn_list'])
        
        self.action_insn_length = dict()
        for action in self.valid_action[contract_name]:
            self.action_insn_length[action] = 0
            for method_name in self.valid_action[contract_name][action]:
                self.action_insn_length[action] += self.method_insn_length[method_name]

        self.action_choices = dict()
        self.action_choices[contract_name] = list()
        self.limit_action = np.ones(self.action_size)
        for action in self.valid_action[contract_name]:
            if self.valid_action[contract_name][action]:
                self.action_choices[contract_name].append(action)
                self.limit_action[action] = 0

    def classification_by_pattern(self, contract_abi):
        classification_dict = dict()
        method_name_to_index = dict()
        index_to_method_name = dict()
        for name in classification_list:
            classification_dict[name] = list()
        for index, name in enumerate(contract_abi['methods']):
            method = contract_abi['methods'][name]
            method_name_to_index[name] = index
            index_to_method_name[index] = name
            if method['row_bow'][SELFDESTRUCT] > 0:
                classification_dict['selfdestruct'].append(name)
            elif method['payable'] == True:
                if method['row_bow'][CALL] > 0 or method['row_bow'][DELEGATECALL] > 0 or method['row_bow'][STATICCALL] > 0:
                    classification_dict['pay-call'].append(name)
                else:
                    classification_dict['pay-nocall'].append(name)
            else:
                if method['row_bow'][CALL] > 0 or method['row_bow'][DELEGATECALL] > 0 or method['row_bow'][STATICCALL] > 0:
                    classification_dict['nopay-call'].append(name)
                else:
                    if method['row_bow'][SSTORE] > 0:
                        classification_dict['nopay-nocall-store'].append(name)

        valid_action = dict()
        for index, name in enumerate(classification_list):
            valid_action[index] = classification_dict[name]

        return valid_action, method_name_to_index, index_to_method_name

    def reset_dqn_state(self):
        self.action_count_array = np.zeros(self.action_size)
        self.tx_count_dqn = 0
        self.trace_bow_accumulative = select_trace_opcode([0 for _ in range(256)])
        self.action_trace = list()
        self.epi_iter += 1
        self.agent_action_count_array = np.zeros(self.action_size)
        
        self.int_agent_action_count_array = np.zeros(255)
        self.uint_agent_action_count_array = np.zeros(257)
        self.bool_agent_action_count_array = np.zeros(2)
        self.addr_agent_action_count_array = np.zeros(len(self.addresses))
        self.byte_agent_action_count_array = np.zeros(256)

    def reset(self):
        self.execution.jump_state(0)

    def load_model(self):
        # load reinforcement model
        self.agent.load(self.args.rl_model)

        # load imitation model
        load_dir = self.args.model
        self.scaler = joblib.load(os.path.join(load_dir, 'scaler.pkl'))
        if use_cuda == 'cuda':
            self.compress_net.load_state_dict(torch.load(os.path.join(load_dir, 'net.pt')))
        else:
            self.compress_net.load_state_dict(torch.load(os.path.join(load_dir, 'net.pt'), map_location='cpu'))
        self.compress_net.eval()

    def calc_method_features(self, contract_name, method_features, scale=False):
        num_methods = len(self.method_names[contract_name])
        features = np.zeros((num_methods, self.raw_feature_size))
        for i, method in enumerate(self.method_names[contract_name]):
            method_w2v = self.nlp.embed_method(method)
            method_feats = np.concatenate([np.array(method_features[method]), method_w2v], axis=0)
            features[i, :self.raw_feature_size] = method_feats[:self.raw_feature_size]
        if scale:
            features = self.scaler.transform(features)
        return features

    # TODO
    def step(self, tx, obs):
        logger = self.execution.commit_tx(tx)

        self.tx_count_dqn += 1
        old_insn_coverage, old_block_coverage = obs.stat.get_coverage(tx.contract)
        destruct, executed_insn_coverage, executed_block_coverage = obs.update(logger, False)
        new_insn_coverage, new_block_coverage = obs.stat.get_coverage(tx.contract)

        reward = ((new_insn_coverage - old_insn_coverage) + (new_block_coverage - old_block_coverage))
        # print(old_insn_coverage, old_block_coverage, new_insn_coverage, new_block_coverage, reward)
        x_state, x_method, contract = self.compute_state(obs)

        return x_state, reward, np.float(destruct), x_method, contract

    def select_tx(self, x_state, x_method, contract, obs, hiddens=[None, None, None, None, None, None], frandom=False, episole=0.3):
        self.int_actions = []
        self.uint_actions = []
        self.bool_actions = []
        self.addr_actions = []
        self.byte_actions = []
        
        # choose method
        r = random.random()
        if r >= 0.2:
            self.slice_size = random.randint(1, 5)
        else:
            self.slice_size = None
        address = contract.addresses[0]

        action, new_hidden_0 = self.agent.choose_action(x_state, self.action_choices[contract.name], self.limit_action, hidden=hiddens[0], episole=episole, agent_action_count_array=self.agent_action_count_array)

        self.action_count_array[action] += 1
        self.action_trace.append(action)
        
        
        pred_f = np.random.choice(self.valid_action[contract.name][action])
        pred_f = self.method_name_to_index[pred_f]

        pred_sender = np.random.choice(len(self.addr_map))
        pred_amount = np.random.choice(len(self.amounts))

        method = contract.abi.methods[pred_f]

        attacker_indices = self.account_manager.attacker_indices
        if np.random.random() < len(attacker_indices) / len(self.account_manager.accounts):
            sender = int(np.random.choice(attacker_indices))
        else:
            sender = pred_sender

        self.last_method[contract.name] = pred_f
        
        new_state, new_method, new_contract = self.compute_state(obs)
        
      
        arguments, _, _, new_hiddens_1_5 = self._select_arguments(contract, method, sender, obs, new_state, new_method[pred_f], hiddens, episole)
        amount = self._select_amount(contract, method, sender, obs, pred_amount)
        timestamp = self._select_timestamp(obs)
        
        new_hiddens = [new_hidden_0] + new_hiddens_1_5
        
        arg_actions = [self.int_actions, self.uint_actions, self.bool_actions, self.addr_actions, self.byte_actions]


        tx = Tx(self, contract.name, address, method.name, bytes(), arguments, amount, sender, timestamp, True)
        # print("Tx: ", method.name,arguments,amount,sender)
        return tx, action, new_hiddens, arg_actions

    def compute_state(self, obs):
        contract = self._select_contract()
        address = contract.addresses[0]

        # deal with the feature of methods
        method_feats = {}
        for m in contract.abi.methods:
            self.method_bows[m.name] = m.bow
        for method, feats in obs.record_manager.get_method_features(contract.name).items():
            method_feats[method] = feats + self.method_bows[method]
                
        if contract.name not in self.last_method:
            last_method_feature = np.zeros(self.feature_size)
            self.method_names[contract.name] = [m.name for m in contract.abi.methods]
        else:
            with torch.no_grad():
                last_method_feature = self.calc_method_features(contract.name, method_feats, True)
                # print(last_method_feature)
                last_method_feature = torch.from_numpy(last_method_feature[self.last_method[contract.name]]).float().to(device)
                last_method_feature = self.compress_net.compress_features(last_method_feature).cpu().numpy()

        x_state = self.action_count_array
        self.trace_bow_accumulative += select_trace_opcode(obs.all_trace_bow)
        trace_op_bow = self.trace_bow_accumulative/self.trace_bow_accumulative.sum() if self.trace_bow_accumulative.sum() > 0 else self.trace_bow_accumulative

        x_state = np.hstack((x_state, last_method_feature, trace_op_bow, np.array(obs.stat.get_coverage(contract.name))))

        x_method = dict()
        for index, feats in enumerate(method_feats.values()):
            x_method[index] = np.array(feats)

        return x_state, x_method, contract

    def _select_contract(self):
        contract_name = random.choice(self.contract_manager.fuzz_contract_names)
        return self.contract_manager[contract_name]

    def _select_sender(self):
        return random.choice(range(0, len(self.account_manager.accounts)))


    def _select_amount(self, contract, method, sender, obs, pred_amount=None):
        if sender in self.account_manager.attacker_indices:
            return 0

        if self.contract_manager.is_payable(contract.name, method.name):
            if pred_amount is None:
                amount = random.randint(0, self.account_manager[sender].amount)
            else:
                amount = self.amounts[pred_amount]
            return amount
        else:
            return 0

    def _select_arguments(self, contract, method, sender, obs, x_state, x_method, hiddens, episole):
        arguments, addr_args, int_args = [], [], []
        new_hidden_1, new_hidden_2, new_hidden_3, new_hidden_4, new_hidden_5 = None, None, None, None, None
        arr_hiddens = [None, None, None, None, None]
        
        for arg in method.inputs:
            t = arg.evm_type.t
            if t == SolType.IntTy or t == SolType.UintTy:
                if t == SolType.IntTy:
                    value, new_hidden_1 = self._select_int(contract, method, arg.evm_type.size, obs, None, x_state, hiddens[1], episole)
                    arguments.append(value)
                elif t == SolType.UintTy:
                    value, new_hidden_2 = self._select_uint(contract, method, arg.evm_type.size, obs, None, x_state, hiddens[2], episole)
                    arguments.append(value)
            elif t == SolType.BoolTy:
                value, new_hidden_3 = self._select_bool(x_state, hiddens[3], episole)
                arguments.append(value)
            elif t == SolType.StringTy:
                arguments.append(self._select_string(obs))
            elif t == SolType.SliceTy:
                arg, arr_hiddens = self._select_slice(contract, method, sender, arg.evm_type.elem, obs, x_state, hiddens, episole)
                arguments.append(arg)
            elif t == SolType.ArrayTy:
                arg, arr_hiddens = self._select_array(contract, method, sender, arg.evm_type.size, arg.evm_type.elem, obs, x_state, hiddens, episole)
                arguments.append(arg)
            elif t == SolType.AddressTy:
                # TODO select address
                value, new_hidden_4 = self._select_address(sender, x_state, hiddens[4], episole)
                arguments.append(value)
            elif t == SolType.FixedBytesTy:
                value, new_hidden_5 = self._select_fixed_bytes(arg.evm_type.size, obs, x_state, hiddens[5], episole)
                arguments.append(value)
            elif t == SolType.BytesTy:
                value, new_hidden_5 = self._select_bytes(obs, x_state, hiddens[5], episole)
                arguments.append(value)
            else:
                assert False, 'type {} not supported'.format(t)
        
        if arr_hiddens[0] is not None:
            new_hidden_1 = arr_hiddens[0]
        if arr_hiddens[1] is not None:
            new_hidden_2 = arr_hiddens[1]
        if arr_hiddens[2] is not None:
            new_hidden_3 = arr_hiddens[2]
        if arr_hiddens[3] is not None:
            new_hidden_4 = arr_hiddens[3]
        if arr_hiddens[4] is not None:
            new_hidden_5 = arr_hiddens[4]

        return arguments, addr_args, int_args, [new_hidden_1, new_hidden_2, new_hidden_3, new_hidden_4, new_hidden_5]

    def _select_int(self, contract, method, size, obs, chosen_int, x_state, hidden, episole):
        limit_action = np.zeros(255)
        limit_action[size-1::] = 1
        choices = [i for i in range(size-1)]
        action, new_hidden = self.int_agent.choose_action(x_state, choices, limit_action, hidden=hidden, episole=episole, agent_action_count_array=self.uint_agent_action_count_array)
        
        shift_table = [i for i in range(-size+1, size)]
        
        shift_1 = shift_table[action*2]
        shift_2 = shift_table[action*2+2] 
        rand_1 = 1<<shift_1 if shift_1 > 0 else -1<<(-shift_1)
        rand_2 = 1<<shift_2 if shift_2 > 0 else -1<<(-shift_2)
        
        value = random.randint(rand_1, rand_2-1)
        
        # s = random.random()
        # if s < 0.9:
        #     value = random.choice(self.int_values_frequent)
        # elif s < 0.98:
        #     value = random.choice(self.int_values_unfrequent)
        # else:
        #     p = 1 << (size - 1)
        #     return random.randint(-p, p-1)

        # value &= ((1 << size) - 1)
        # if value & (1 << (size - 1)):
        #     value -= (1 << size)
        self.int_actions.append(action)
        return value, new_hidden

    def _select_uint(self, contract, method, size, obs, chosen_int, x_state, hidden, episole):
        limit_action = np.zeros(257)
        limit_action[size+1::] = 1
        choices = [i for i in range(size+1)]
        action, new_hidden = self.uint_agent.choose_action(x_state, choices, limit_action, hidden=hidden, episole=episole, agent_action_count_array=self.uint_agent_action_count_array)
        value = 0 if action == 0 else random.randint(1<<(int(action)-1), (1<<int(action))-1)
        # print(action, value)     
      
        # s = random.random()
        # if s < 0.9:
        #     value = random.choice(self.int_values_frequent)
        # elif s < 0.98:
        #     value = random.choice(self.int_values_unfrequent)
        # else:
        #     p = 1 << size
        #     return random.randint(0, p-1)
        self.uint_actions.append(action)
        return value, new_hidden

    def _select_address(self, sender, x_state, hidden, episole):
        normal_users = [addr for addr in self.addresses if addr not in self.account_manager.attacker_addresses]
        normal_choices = [i for i in range(len(normal_users))]
        choices = [i for i in range(len(self.addresses))]
        limit_attacker = np.zeros(len(self.addresses))
        no_limit = np.zeros(len(self.addresses))
        for addr in self.account_manager.attacker_addresses:
            if addr in self.account_manager.attacker_addresses:
                limit_attacker[self.addresses.index(addr)] = 1
        if sender in self.account_manager.attacker_indices:
            action, new_hidden = self.addr_agent.choose_action(x_state, choices, no_limit, hidden=hidden, episole=episole, agent_action_count_array=self.addr_agent_action_count_array)
        else:
            action, new_hidden = self.addr_agent.choose_action(x_state, normal_choices, limit_attacker, hidden=hidden, episole=episole, agent_action_count_array=self.addr_agent_action_count_array)
        self.addr_actions.append(action)
        return self.addresses[action], new_hidden

    def _select_bool(self, x_state, hidden, episole):
        choices = [0, 1]
        limit_action = np.zeros(2)
        action, new_hidden = self.bool_agent.choose_action(x_state, choices, limit_action, hidden=hidden, episole=episole, agent_action_count_array=self.uint_agent_action_count_array)
        self.bool_actions.append(action), new_hidden
        
        return action

    def _select_string(self, obs):
        bs = []
        size = random.randint(0, 40)
        for _ in range(size):
            bs.append(random.randint(1, 127))
        return bytearray(bs).decode('ascii')

    def _select_slice(self, contract, method, sender, typ, obs, x_state, hidden, episole):
        if self.slice_size is None:
            size = random.randint(1, 15)
        else:
            size = self.slice_size
        return self._select_array(contract, method, sender, size, typ, obs, x_state, hidden, episole)

    def _select_array(self, contract, method, sender, size, typ, obs, x_state, hiddens, episole):
        t = typ.t
        arr = []

        new_hidden_1, new_hidden_2, new_hidden_3, new_hidden_4, new_hidden_5 = None, None, None, None, None
        arr_hiddens = [None, None, None, None, None]

        for _ in range(size):

            if t == SolType.IntTy or t == SolType.UintTy:
                if t == SolType.IntTy:
                    value, new_hidden_1 = self._select_int(contract, method, arg.evm_type.size, obs, None, x_state, hiddens[1], episole)
                    arr.append(value)
                elif t == SolType.UintTy:
                    value, new_hidden_2 = self._select_uint(contract, method, arg.evm_type.size, obs, None, x_state, hiddens[2], episole)
                    arr.append(value)
            elif t == SolType.BoolTy:
                value, new_hidden_3 = self._select_bool(x_state, hiddens[3], episole)
                arr.append(value)
            elif t == SolType.StringTy:
                arr.append(self._select_string(obs))
            elif t == SolType.SliceTy:
                arg, arr_hiddens = self._select_slice(contract, method, sender, arg.evm_type.elem, obs, x_state, hiddens, episole)
                arr.append(arg)
            elif t == SolType.ArrayTy:
                arg, arr_hiddens = self._select_array(contract, method, sender, arg.evm_type.size, arg.evm_type.elem, obs, x_state, hiddens, episole)
                arr.append(arg)
            elif t == SolType.AddressTy:
                # TODO select address
                value, new_hidden_4 = self._select_address(sender, x_state, hiddens[4], episole)
                arr.append(value)
            elif t == SolType.FixedBytesTy:
                value, new_hidden_5 = self._select_fixed_bytes(arg.evm_type.size, obs, x_state, hiddens[5], episole)
                arr.append(value)
            elif t == SolType.BytesTy:
                value, new_hidden_5 = self._select_bytes(obs, x_state, hiddens[5], episole)
                arr.append(value)
            else:
                assert False, 'type {} not supported'.format(t)
                
        if arr_hiddens[0] is not None:
            new_hidden_1 = arr_hiddens[0]
        if arr_hiddens[1] is not None:
            new_hidden_2 = arr_hiddens[1]
        if arr_hiddens[2] is not None:
            new_hidden_3 = arr_hiddens[2]
        if arr_hiddens[3] is not None:
            new_hidden_4 = arr_hiddens[3]
        if arr_hiddens[4] is not None:
            new_hidden_5 = arr_hiddens[4]

        return arr, [new_hidden_1, new_hidden_2, new_hidden_3, new_hidden_4, new_hidden_5]

    def _select_fixed_bytes(self, size, obs, x_state, hidden, episole):
        bs = []
        for _ in range(size):
            choices = [i for i in range(256)]
            limit_action = np.zeros(256)
            action, new_hidden = self.bool_agent.choose_action(x_state, choices, limit_action, hidden=hidden, episole=episole, agent_action_count_array=self.uint_agent_action_count_array)
            bs.append(action)
        self.byte_actions.append(bs), new_hidden
        return bs

    def _select_bytes(self, obs, x_state, hidden, episole):
        size = random.randint(1, 15)
        return self._select_fixed_bytes(size, obs, x_state, hidden, episole)