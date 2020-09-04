import tensorflow as tf
import baselines.common.tf_util as U

tf.reset_default_graph()

def build_train(q_func, batch_norm, update_step, obs_dim, output_dim, optimizer, grad_norm_clipping=None, 
    scope="deepq", reuse=None):
    '''
    Building training ops - train, update_target, debug_dict.
    '''
  
    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        states_ph = tf.placeholder(tf.float32, [None, obs_dim], name="obs_t")
        phase_ph = tf.placeholder(tf.bool, name="is_training")
        NN_target = tf.placeholder(tf.float32, [None, output_dim], name="target_t")
        
        # q network evaluation
        NN_t = q_func(states_ph, output_dim, scope="q_func", phase = phase_ph, batch_norm = batch_norm)        
        q_t = tf.cumsum(NN_t, axis = 1)
        NN_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")
        
        # target q network evalution
        NN_tp_target = q_func(states_ph, output_dim, scope="target_q_func", phase = phase_ph, batch_norm = batch_norm)
        q_tp_target = tf.cumsum(NN_tp_target, axis = 1)
        target_NN_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/target_q_func")

           
        # compute the error (potentially clipped)
        td_error = NN_t - tf.stop_gradient(NN_target)
        mean_error = tf.nn.l2_loss(td_error)
        
        
        # Train op:
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            optimize_expr = optimizer.minimize(mean_error, var_list=NN_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(NN_func_vars, key=lambda v: v.name),
                                   sorted(target_NN_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)


        # Create callable functions
        train = U.function( inputs=[states_ph, NN_target, phase_ph],
                            outputs=td_error,
                            updates=[optimize_expr] )
        
        update_target = U.function([], [], updates=[update_target_expr])


        NN_values = U.function([states_ph, phase_ph], NN_t)
        q_values = U.function([states_ph, phase_ph], q_t)
                
        NN_values_target = U.function([states_ph, phase_ph], NN_tp_target)
        q_values_target = U.function([states_ph, phase_ph], q_tp_target)
        
        
        return train, update_target, {'NN_train': NN_values, 'q_train': q_values, 
                                      'NN_target': NN_values_target, 'q_target': q_values_target,
                                     }
                                


