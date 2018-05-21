library(gym)
library(profvis)

debugSource("DQNAgent.R")

# Parameters.
EPISODES <- 100
BATCH_SIZE <- 64
LEARNING_RATE <- 0.001
GAMMA <- 0.99
EPSILON_START <- 1.0
EPSILON_MIN <- 0.01
EPSILON_DECAY <- 0.999

if (exists("memory"))
  rm(memory)

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
  score <- 1
  
  while (TRUE) {
    
    score = score + 1
    thisAction <- act(state = state,
                  model = model,
                  epsilon = epsilon,
                  availableActions = actionSpace)
    
    nextState <- env_step(
      client,
      instanceID,
      thisAction,
      render = FALSE
    )
    
    reward <- ifelse(nextState$done & score < 499, -100, nextState$reward)
    memory <- remember(state, thisAction, reward, unlist(nextState$observation), nextState$done, memory)
    
    if (epsilon > EPSILON_MIN) {
      epsilon = epsilon * EPSILON_DECAY
    }
    
    state <- unlist(nextState$observation)
    
    if (nrow(memory) > BATCH_SIZE) {
      
      replay(
        memory = memory, 
        model = model, 
        targetModel = targetModel,
        batchSize = BATCH_SIZE, 
        gamma = GAMMA
      )
    }
    
    if (nextState$done) {
      updateTargetModel(model, targetModel)
      print(paste("Episode:", i, "/", EPISODES, "Score:", score, "Epsilon:", epsilon))
      break;
    }
  }
}


# Dump result info to disk
env_monitor_close(client, instanceID)
