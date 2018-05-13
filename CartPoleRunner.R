library(gym)

source("DQNAgent.R")

# Parameters.
EPISODES <- 5000
BATCH_SIZE <- 32
LEARNING_RATE <- 0.001
GAMMA <- 0.95
EPSILON <- 1.0
EPSILON_MIN <- 0.01
EPSILON_DECAY <- 0.99

# Open AI Gym Environment.
remoteURL <- "http://127.0.0.1:5000"
client <- create_GymClient(remoteURL)

# Cart Pole
environmentID <- "CartPole-v0"
instanceID <- env_create(client, environmentID)

actionSpace <- env_action_space_info(client, instanceID)
stateSpace <- env_observation_space_info(client, instanceID)$shape[[1]]

# Build the DL models for Q estimation.
model <- buildModel(stateSize = stateSpace, actionSize = actionSpace, learningRate = LEARNING_RATE)
targetModel <- buildModel(stateSize = stateSpace, actionSize = actionSpace, learningRate = LEARNING_RATE)

# Dump result info to disk
env_monitor_close(client, instanceID)
