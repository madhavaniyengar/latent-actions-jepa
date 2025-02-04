import torch
import os
import numpy as np
import cv2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transformations=['resize'], action_representation='action'):
        self.data_path = data_path
        self.trajectories = os.listdir(data_path)
        self.transformations = transformations
        self.action_representation = action_representation
        
        self.actions = self.load_actions()
        self.observations = self.load_observations()
            
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], idx
    
    def load_actions(self):
        actions = []
        for trajectory in self.trajectories:
            actions_path = os.path.join(self.data_path, trajectory, self.action_representation)
            actions_files = os.listdir(actions_path)
            sorted_actions_files = sorted(actions_files, key=lambda x: int(x.split('.')[0]))
            for file in sorted_actions_files:
                action = np.load(str(os.path.join(actions_path, file)))
                action = torch.tensor(action, dtype=torch.float32)
                actions.append(action)    
        return actions
    
    def load_observations(self):
        video_observations = []
        for trajectory in self.trajectories:
            observations_path = os.path.join(self.data_path, trajectory, 'env_visual')
            observations_files = os.listdir(observations_path)
            sorted_observations_files = sorted(observations_files, key=lambda x: int(x.split('.')[0]))
            raw_observations = []
            
            for file in sorted_observations_files:
                raw_observation = np.load(os.path.join(observations_path, file))
                if 'resize' in self.transformations:
                    raw_observation = cv2.resize(raw_observation, (224, 224)) # resize the image to 224x224
                raw_observation = torch.tensor(raw_observation, dtype=torch.float32)
                raw_observations.append(raw_observation)
            
            for idx, raw_observation in enumerate(raw_observations):
                start_idx = 0 if idx < 15 else idx - 15
                if idx == 0:
                    video_obs = raw_observations[0].unsqueeze(0)
                else:
                    video_obs = torch.stack(raw_observations[start_idx:idx+1], dim=0)
                if idx < 15:
                    pad_frames = 15 - idx 
                    pad_tensor = torch.zeros(pad_frames, video_obs.shape[1], video_obs.shape[2], video_obs.shape[3])
                    video_obs = torch.cat([pad_tensor, video_obs], dim=0)
                video_observations.append(video_obs)
        return video_observations

if __name__ == '__main__':
    dataset = Dataset('/home/madhavan/jepa/policy/data/pick_mug')
    print(len(dataset))
    print(dataset[0][0].shape, dataset[0][1].shape)
    print(dataset[0][0])