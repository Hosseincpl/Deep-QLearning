import MetaTrader5 as mt5
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95 
        self.epsilon = 1.0  
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
    def _build_model(self):
        # Neural Network for Deep Q-Learning
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    # Connect to MetaTrader 5
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()
    # Define the trading parameters and assets
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_H1
    window_size = 10
    state_size = window_size + 1
    buy = 0
    sell = 1
    hold = 2
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 1000
    # Start the training loop
    for episode in range(episodes):
        state = get_state(symbol, timeframe, window_size)
        done = False
        total_profit = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, profit = perform_action(symbol, timeframe, action)
            total_profit += profit
            agent.remember(state, action, reward, next_state, done)
            state = next_state
        print("episode: {}/{}, profit: {}".format(episode + 1, episodes, total_profit))
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    # Disconnect from MetaTrader 5
    mt5.shutdown()
