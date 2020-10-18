import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        print(type(observation))
        print(observation)
        print(observation.shape)
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        qa_values = self.critic.qa_values(observation)
        #print("qa_values.shape", qa_values.shape)
        #print(type(qa_values))
        #print(qa_values)
        print(type(qa_values))
        print(qa_values)
        print(qa_values.shape)

        action = np.array([np.argmax(qa_values)])
        
        print(action)
        #print(action)
        #squeezed = action.squeeze()
        #print(squeezed)
        return action.squeeze()