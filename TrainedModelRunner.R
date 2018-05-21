library(gym)

source("DQNAgent.R")

# Open AI Gym Environment.
remoteURL <- "http://127.0.0.1:5000"
client <- create_GymClient(remoteURL)

# Cart Pole
environmentID <- "CartPole-v0"
instanceID <- env_create(client, environmentID)

actionSpace <- env_action_space_info(client, instanceID)
stateSpace <- env_observation_space_info(client, instanceID)$shape[[1]]

state <- unlist(env_reset(client, instanceID))
score <- 0

while (TRUE) {
  
  score <- score + 1
  
  thisAction <- act(state = state,
                    model = targetModel,
                    epsilon = 0,
                    availableActions = actionSpace)
  
  state <- env_step(
    client,
    instanceID,
    thisAction,
    render = TRUE
  )
  
  if (state$done) {
    print(paste("Score:", score))
    break;
  }
  
  Sys.sleep(0.02)
  state <- unlist(state$observation)
}

Sys.sleep(10)
env_monitor_close(client, instanceID)
