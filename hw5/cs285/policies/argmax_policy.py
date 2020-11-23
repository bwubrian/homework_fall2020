import numpy as np
import pdb


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def set_critic(self, critic):
        self.critic = critic

    def get_action(self, obs):
        # MJ: changed the dimension check to a 3
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        # TODO: get this from hw3
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        q_values = self.critic.qa_values(observation)
        action = q_values.argmax(-1)
        #print("get_action q_values.shape", q_values.shape)
        #print("action", repr(action))
        return action[0]

    def get_sampled_action(self, obs):
        # MJ: changed the dimension check to a 3
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        # TODO: get this from hw3
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        q_values = self.critic.qa_values(observation)
        #print("q_values.shape", q_values.shape)
        exp_q_values = np.exp(q_values)
        exp_sum_q_values = exp_q_values / np.sum(exp_q_values)
        #print("exp_sum_q_values", exp_sum_q_values)
        exp_sum_q_values = exp_sum_q_values.flatten()
        action = np.random.choice(np.arange(q_values.shape[1]), p=exp_sum_q_values)
        return action

    ####################################
    ####################################