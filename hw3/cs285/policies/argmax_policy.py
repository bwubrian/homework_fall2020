import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        qa_values = self.critic.qa_values(observation)
        print("qa_values.shape", qa_values.shape)
        print(type(qa_values))
        action = np.argmax(qa_values)
        return action.squeeze()