import time
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.layers as layers
import baselines.common.tf_util as U
from replay_buffer_Q import ReplayBuffer

### Custom DQN Functions:
import Q_utils as DQN

### Custom Environment
from env_rnn_mkt import Environment
Dir = '/Users/muyewang/Google Drive/CBS_Reseach/Optimal Execution_2020/RL_Implementation/DQN_Implementation/Data/episodes_rnn_short/'
#Dir = '../../Data/episodes_rnn_short/'

### Reset TF:
tf.reset_default_graph()
sess = tf.InteractiveSession()

layer_initialize_seed = 0
############################################### RL Algorithm Class ###############################################
class RL_algo():
    def __init__(self, Dir, feature, double_q = True, rnn_seq = 5):
        # Hyper-Parameters:
        self.initial_step = 5000
        self.train_step = 10000   
        self.test_freq = 100
        self.batch_size = 1024
        self.learning_rate = 3e-5
        self.copy_freq = 250

        self.update_step = 1

        # Feature Parameters:
        self.rnn_states = 9
        self.rnn_seq = rnn_seq
        self.feature = feature
        self.double_q = double_q
        
        # RL-related objects
        self.replay_buffer = ReplayBuffer(50000, self.update_step)                   
        self.env = Environment(Dir, feature, rnn_seq)
        self.N = self.env.N
        self.sess = U.make_session()
        
        # Functions
        self.train = None
        self.update_target = None
        self.debug = None
        
        # Testing Results
        self.epi_rews_train = []
        self.epi_rews_test = []
        self.epi_rews_validate = []
        self.target_time = []
        self.gradient_time = []
        
        
    def build_train(self, batch_norm):
        '''
        Initialize build ops
        '''
        if self.feature == 'RNN Feature':
            model = self.rnn_model
        else:
            model = self.nn_model
        
        train, update_target, debug = DQN.build_train(
            q_func = model,
            batch_norm = batch_norm,
            update_step = self.update_step,
            obs_dim = self.env.observation_space.shape[0]-1,
            output_dim = self.N, 
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate),
            )
        self.train = train
        self.update_target = update_target
        self.debug = debug
        print("-------------------------------------------")
        print("Batch Normalization: ", batch_norm)
    
    def initialize_session(self):
        '''
        Initialize tf session
        '''
        U.initialize()
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)
        self.update_target()

    
    ########### Training Function ###########
    def fill_replay_buffer(self):
        '''
        Pre-fill replay buffer with experiences
        '''
        print("-------------------------------------------")
        print("Filling Buffers")
        print("-------------------------------------------")
        for t in range(self.initial_step):
            begin_state, rews_trajectory, last_state, states_trajectory = self.env.sample_trajectory(self.update_step)
            self.replay_buffer.add(rews_trajectory, states_trajectory)
 
    
    def train_rl(self):
        '''
        Repeat the following each step:
        1. Sample a single experience from data, add to the buffer
        2. Call Recurrent action
        '''
        print("Training Starts......")
        ### Add another trajectory to replay buffer
        for t in itertools.count():
            begin_state, rews_trajectory, last_state, states_trajectory = self.env.sample_trajectory(self.update_step)
            self.replay_buffer.add(rews_trajectory, states_trajectory)
 
            if t > self.train_step:
                self.sess.close()
                break
            else:
                self.recurrent_action(t)
            

    def recurrent_action(self, t):
        # Test RMSE and Episode Rewards:
        if t % self.test_freq == 0:
            print("-------------------------------------------")
            print("Testing - Step: ", t)
            rew_per_episode_mean_1 = self.test_episode_performance(scope = 'training')
            self.epi_rews_train.append(rew_per_episode_mean_1)
            print("-------------------------------------------")
            rew_per_episode_mean_2 = self.test_episode_performance(scope = 'testing')
            self.epi_rews_test.append(rew_per_episode_mean_2)
            print("-------------------------------------------")
            rew_per_episode_mean_3 = self.test_episode_performance(scope = 'validate')
            self.epi_rews_validate.append(rew_per_episode_mean_3)
            print("-------------------------------------------")

        # Training - Gradient Step
        rews_trajectory, states_trajectory = self.replay_buffer.sample(self.batch_size)
        
        start_time = time.time()
        
        if self.double_q:
            Q_action = np.stack([self.debug['q_train'](states_trajectory[:,i,:]) for i in range(1, self.update_step+1)], axis = 1)
            Q_value = self.debug['q_target'](states_trajectory[:,-1,:])
            Q_value = np.concatenate([np.zeros_like(Q_action[:,0,:Q_action.shape[1]]), Q_value[:,:self.N-Q_action.shape[1]]], axis = 1)

        else:
            Q_action = np.stack([self.debug['q_target'](states_trajectory[:,i,:]) for i in range(1, self.update_step+1)], axis = 1)
            Q_value = np.concatenate([np.zeros_like(Q_action[:,0,:Q_action.shape[1]]), Q_action[:,-1,:self.N-Q_action.shape[1]]], axis = 1)
        
        # Calculate Indicators:        
        indicator_values = [np.concatenate([np.zeros_like(Q_action[:,0:1,:i+1]), Q_action[:,i:i+1,:self.N-i-1]], axis = 2) for i in range(Q_action.shape[1])]
        indicator_values_arr = np.concatenate(indicator_values, axis = 1)
        indicators = (indicator_values_arr > 0).astype(np.int32)
        indicators_cumprod = np.cumprod(indicators, axis = 1)
        indicators_final = np.concatenate([np.ones_like(indicators_cumprod[:,0:1,:]), indicators_cumprod], axis = 1)
        
        # Calculate Target Values:        
        rews_trajectory_stack = np.repeat(np.expand_dims(rews_trajectory, axis =2), self.N, axis = 2)
        values_stack = np.concatenate([rews_trajectory_stack, np.expand_dims(Q_value, axis = 1) ], axis = 1)
        Q_target = np.multiply(values_stack, indicators_final).sum(axis = 1)
        NN_target = np.concatenate([Q_target[:, 0:1], np.diff(Q_target, axis = 1)], axis = 1)
        
        # time the training operations:
        end_time = time.time()
        time_calculate_target = end_time - start_time
        self.target_time.append(time_calculate_target)
        
        start_time = time.time()
        self.train(states_trajectory[:,0,:], NN_target, True)
        end_time = time.time()
        time_gradient_step = end_time - start_time
        self.gradient_time.append(time_gradient_step)
        
        # Update target network periodically, display learned function
        if t % self.copy_freq == 0:   
            self.update_target()
            df_NN = rl.test_learned_q()
            print("Copy Action Taken")
            print("Train-net At step: ", t)
            print(df_NN)
              
           
    def test_episode_performance(self, scope = 'testing'):
        _, price_change_cum = self.env.all_feature_scope(time_index = 0, scope = scope)
        price_diff = np.concatenate([np.expand_dims(price_change_cum[:,0], axis = 1), np.diff(price_change_cum, axis = 1)], axis = 1)
        action_list = []
        for i in range(self.N):      
            obs_feature, _ = self.env.all_feature_scope(time_index = i, scope = scope)
            q_value = self.debug['q_train'](obs_feature, False)
            action_est = ( q_value[:, self.N-1-i ] < 0 ).astype(int) 
            action_list.append(action_est)
            
        action_arr = np.array(action_list).T
        action_reward = [ action_arr[:,:i].max(axis = 1) for i in range(1, self.N+1)]
        action_reward_arr = np.array(action_reward).T
        reward_arr = 1 - action_reward_arr
        rewards = np.multiply( price_diff, reward_arr )
        
        rew_per_episode_mean = rewards.sum(axis=1).mean()
        rew_per_episode_sem = rewards.sum(axis=1).std() / np.sqrt(rewards.shape[0])
        
        print("Testing Performance (" +scope+"): ")
        print("Episode Reward - Mean:", round(rew_per_episode_mean, 3))
        print("Episode Reward - S.E.:", round(rew_per_episode_sem, 3))
        return rew_per_episode_mean
        
    
    ######## Neural Networks ########
    def nn_model(self, inpt, output_dim, scope,  phase, batch_norm,  reuse=False):
        def dense_linear(x, size):   
            return layers.fully_connected(x, size, activation_fn=None, 
                                          weights_initializer=tf.contrib.layers.xavier_initializer(seed=layer_initialize_seed))
        def dense_relu(x, size):
            return layers.fully_connected(x, size, activation_fn=tf.nn.relu, 
                                          weights_initializer=tf.contrib.layers.xavier_initializer(seed=layer_initialize_seed))
        def dense_batch_relu(x, size):
            h1 = layers.fully_connected(x, num_outputs = size, activation_fn=None, 
                                        weights_initializer=tf.contrib.layers.xavier_initializer(seed=layer_initialize_seed))
            h2 = layers.batch_norm(h1, center=True, scale=True, is_training=phase)
            return tf.nn.relu(h2)
        
        ## Neural Network Topology:
        with tf.variable_scope(scope, reuse=reuse):
            out = inpt
            for i in range(5):
                if batch_norm:   out = dense_batch_relu(out, 64)
                else:            out = dense_relu(out, 64)
            out = dense_linear(out, output_dim)
            # Apply Exponential Last Layer:
            out = tf.concat([tf.expand_dims(out[:, 0], axis = 1), tf.exp(out[:, 1:])], axis = 1)
        return out
    
    
    def rnn_model(self, inpt, output_dim, scope, phase, batch_norm, reuse=False):
        def dense_linear(x, size):
            return layers.fully_connected(x, size, activation_fn=None)
        
        def dense_relu(x, size):
            return layers.fully_connected(x, size, activation_fn=tf.nn.relu)
            
        def dense_batch_relu(x, size):
            h1 = layers.fully_connected(x, num_outputs = size, activation_fn=None)
            h2 = layers.batch_norm(h1, center=True, scale=True, is_training=phase)
            return tf.nn.relu(h2)
        
        inpt_rnn = tf.reshape(inpt, [tf.shape(inpt)[0], self.rnn_seq, self.rnn_states])   #shape = [batch_size, rnn_seq, rnn_state]
        
        with tf.variable_scope(scope, reuse=reuse):
            inputs_series = tf.split(inpt_rnn, self.rnn_seq, 1)
            inputs_series = [tf.squeeze(ts,axis = 1) for ts in inputs_series]
            cell = tf.contrib.rnn.BasicRNNCell(64)
            states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, dtype =tf.float32 )
            
            out = current_state
            for i in range(5):
                if batch_norm:   out = dense_batch_relu(out, 64)
                else:            out = dense_relu(out, 64)
            out = dense_linear(out, output_dim)
        return out

            

if __name__ == '__main__':
    rl = RL_algo(Dir, 'QI', double_q = True)
    
    print("Update Step: ", rl.update_step)
    print("Initialization: ", layer_initialize_seed)
    
    # Training:
    rl.build_train(batch_norm = False)
    rl.initialize_session()
    df_NN = rl.test_learned_q()
    print("Initial Q_Function")
    print(df_NN.iloc[:5])
    
    rl.fill_replay_buffer()
    rl.train_rl()
    
    # Results:
    result_df = pd.DataFrame( np.array([rl.epi_rews_train, rl.epi_rews_test, rl.epi_rews_validate]).T, 
         columns = ['epi_rews_train', 'epi_rews_test', 'epi_rews_validate'])
    result_df.to_csv(save_dir+'results_'+str(rl.update_step)+'_'+str(layer_initialize_seed)+'.csv')
    
    
     
    
    