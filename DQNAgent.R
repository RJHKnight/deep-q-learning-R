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
MIN_MEMORY_SIZE <- 1000

huberLoss <- function(target, prediction) {
  
  K <- backend()
  
  # calculate the metric
  error = prediction - target
  errorSquared <- K$square(error)
  rootErrorSquared <- K$sqrt(K$constant(1, dtype = "float32") + K$square(error))
  
  metric <- K$mean(rootErrorSquared-1)
  
  return (metric)
}

# Neural Net for Deep-Q learning Model
buildModel <- function(stateSize, actionSize, learningRate) {
 
  model <- keras_model_sequential() 
  
  model %>% 
    layer_dense(units = 24, input_shape = c(stateSize), kernel_initializer = 'he_uniform') %>% 
    layer_activation('relu') %>% 
    layer_dense(units = 24, input_shape = c(stateSize), kernel_initializer = 'he_uniform') %>% 
    layer_activation('relu') %>% 
    layer_dense(units = actionSize, kernel_initializer = 'he_uniform') %>% 
    layer_activation('linear')
  
  model %>% compile(
    loss = loss_mean_squared_error,
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


matrixToList <- function(x) {
  lapply(1:nrow(x), function(i) x[i,])
}

replay <- function(memory, model, targetModel, batchSize, gamma) {
  
  if (nrow(memory) < MIN_MEMORY_SIZE) {
    return ()
  }
  
  thisStateColumns =  grep("state", names(memory), value=T)
  nextStateColumns =  grep("nextState", names(memory), value=T)
  miniBatch <- memory[sample(nrow(memory), batchSize),]
  
  thisState <- as.matrix(miniBatch[,thisStateColumns])
  thisNextState <- as.matrix(miniBatch[, nextStateColumns])
  
  target <- matrix(rep(NA, 2 * batchSize), ncol = 2)
  
  for (i in 1:batchSize) {

     thisFinalState <- miniBatch$finalState[i]
     thisAction <- miniBatch$action[i]
     thisReward <- miniBatch$reward[i]
     
     thisTarget <- model %>%
        predict(matrix(thisState[i,], ncol = 4))
     
     targetNext <- model %>%
       predict(matrix(thisNextState[i,], ncol=4))
     
     targetVal = targetModel %>%
       predict(matrix(thisNextState[i,], ncol=4))
     
     if (thisFinalState) {
       thisTarget[thisAction+1] <- thisReward
     }
     else {
       
       action = which.max(targetNext)
       thisTarget[thisAction+1] = thisReward + (gamma * targetVal[action])
     }
     
     target[i,] <- thisTarget
     
  }
  
  model %>%
    fit(thisState, target, epochs = 1, verbose = FALSE)
  
  return ()
    
}

updateTargetModel <- function(model, targetModel) {
  
  targetModel$set_weights(model$get_weights())
  
  return (targetModel)  
}
  