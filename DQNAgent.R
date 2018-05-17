library(keras)
library(tidyverse)

# TODO: 
# 1) (x) Huber Loss 
# 2) (x) Build model
# 3) Update target model
# 4) (x) act
# 5) replay
# 6) load
# 7) save
# 8) run?
# 9) (x) remember

# Memory:
# 1) State
# 2) Action
# 3) Reward
# 4) NextState
# 5) FinalState

MAX_MEMORY_SIZE <- 2000

huberLoss <- function(target, prediction) {
  
  K <- backend()
  
  # calculate the metric
  error = prediction - target
  errorSquared <- K$square(error)
  rootErrorSquared <- K$sqrt(1+K$square(error))
  
  metric <- K$mean(rootErrorSquared -1, axis = as.integer(-1))
  
  return (metric)
}

# Neural Net for Deep-Q learning Model
buildModel <- function(stateSize, actionSize, learningRate) {
 
  model <- keras_model_sequential() 
  
  model %>% 
    layer_dense(units = 24, input_shape = c(stateSize)) %>% 
    layer_activation('relu') %>% 
    layer_dense(units = actionSize) %>% 
    layer_activation('linear')
  
  model %>% compile(
    loss = loss_mean_absolute_error,
    optimizer = optimizer_adam(lr = learningRate)
    )
  
  return (model)
  
}

act <- function(state, epsilon, model, availableActions) {
  
  randomExploration <- (runif(1) <= epsilon)
  
  if (randomExploration) {
    return (sample(availableActions$n, 1)-1)
  }
  
  actionValues <- model %>%
    predict(matrix(state, nrow=1))
    
  return (which.max(actionValues)-1)
  
}

remember <- function(state, action, reward, nextState, finalState, memory) {
  
  # TODO: Optimize me.
  thisEntry <- data.frame(state = t(state), 
                          action = action,
                          reward = reward,
                          nextState = t(nextState),
                          finalState = finalState)
  
  if (is.null(memory)) {
    return (thisEntry)
  }
  
  if (nrow(memory) >= MAX_MEMORY_SIZE)
    memory <- memory[-1,]
  
  return (rbind(memory, thisEntry))
}


replay <- function(memory, model, targetModel, batchSize, gamma, epsilon, epsilonMin, epsilonDecay) {
  
  miniBatch <- sample(nrow(memory), batchSize)
  
  for (i in 1:batchSize) {
    
     thisStateColumns =  grep("state", names(memory), value=T)
     nextStateColumns =  grep("nextState", names(memory), value=T)

     thisState <- as.matrix(memory[i,thisStateColumns])
     thisNextState <- as.matrix(memory[i, nextStateColumns])
     
     thisFinalState <- memory$finalState[i]
     thisAction <- memory$action[i]
     thisReward <- memory$reward[i]
     
     target <- model %>%
        predict(thisState)
     
     if (thisFinalState) {
       target[thisAction] <- thisReward
     }
     else {
       t <- targetModel %>%
         predict(thisNextState)
       
       target[thisAction+1] = thisReward #+ (gamma * t[which.max(t)])
     }
     
     model %>%
       fit(thisState, target, epochs = 1, verbose = FALSE)
  }
     
  if (epsilon > epsilonMin) {
    epsilon = epsilon * epsilonDecay
  }
  
  return (epsilon)
    
}

updateTargetModel <- function(model, targetModel) {
  
  targetModel$set_weights(model$get_weights())
  
  return (targetModel)  
}
  