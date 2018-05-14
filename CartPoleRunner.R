library(gym)

debugSource("DQNAgent.R")

# Parameters.
EPISODES <- 5000
BATCH_SIZE <- 32
LEARNING_RATE <- 0.001
GAMMA <- 0.95
EPSILON_START <- 1.0
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
model <- buildModel(stateSize = stateSpace, actionSize = actionSpace$n, learningRate = LEARNING_RATE)
targetModel <- buildModel(stateSize = stateSpace, actionSize = actionSpace$n, learningRate = LEARNING_RATE)

# Initialise epsilon and memory.
epsilon <- EPSILON_START
memory <- NULL

for (i in 1:EPISODES) {
   
  state <- unlist(env_reset(client, instanceID))
  
  for (t in 1:500) {
    
    thisAction <- act(state = state,
                  model = model,
                  epsilon = epsilon,
                  availableActions = actionSpace)
    
    nextState <- env_step(
      client,
      instanceID,
      thisAction,
      render = TRUE
    )
    
    reward <- ifelse(nextState$done, -10, nextState$reward)
    memory <- remember(state, thisAction, reward, unlist(nextState$observation), nextState$done, memory)
    state <- unlist(nextState$observation)
    
    if (nextState$done) {
      updateTargetModel(model, targetModel)
      print(paste("Episode:", i, "/", EPISODES, "Score:", t, "Epsilon:", epsilon))
      break;
    }
    
    if (nrow(memory) > BATCH_SIZE) {
      
      epsilon <- replay(
        memory = memory, 
        model = model, 
        targetModel = targetModel, 
        batchSize = BATCH_SIZE, 
        gamma = GAMMA,
        epsilon = epsilon,
        epsilonMin = EPSILON_MIN,
        epsilonDecay = EPSILON_DECAY
      )
    }
  }
}



# Dump result info to disk
env_monitor_close(client, instanceID)
