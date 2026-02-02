import collections, random
import numpy as np

# Structure for the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        # Use a deque as it automatically handles max size
        self.buffer = collections.deque(maxlen=capacity)
        self.batch_size = batch_size

    def store(self, state, action, reward, next_state, done):
        """Stores a transition tuple in the buffer."""
        # Ensure states are NumPy arrays for consistency
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        # Add the experience tuple to the deque
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        """Samples a mini-batch of experiences."""
        # Randomly select indices for the batch
        batch_indices = random.sample(range(len(self.buffer)), self.batch_size)
        # Retrieve the experiences corresponding to the sampled indices
        experiences = [self.buffer[i] for i in batch_indices]

        # Unzip the batch into separate arrays for states, actions, etc.
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert to NumPy arrays for batch processing by the network
        return (np.concatenate(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.concatenate(next_states),
                np.array(dones, dtype=np.uint8))

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)

    def check_length(self):
        return len(self.buffer) > self.batch_size
    