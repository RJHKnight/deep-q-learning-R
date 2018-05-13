library(keras)
library(tidyverse)

EPISODES <- 5000

loss_huber <- function(target, prediction) {
  
  error = prediction - target
  return (mean(sqrt(1+square(error))-1))
}


# Neural Net for Deep-Q learning Model
buildModel <- function(stateSize, actionSize, learningRate) {
 
  model <- keras_model_sequential() 
  
  model %>% 
    layer_dense(units = 24, input_shape = c(stateSize)) %>% 
    layer_activation('relu') %>% 
    layer_dense(units = activationSize) %>% 
    layer_activation('linear')
  
  model %>% compile(
    loss = loss_huber,
    optimizer = optimizer_adam(lr = learningRate)
    )
  
  return (model)
  
}