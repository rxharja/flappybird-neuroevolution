# flappybird-neuroevolution
![GIF of two birds training](https://github.com/rxharja/flappybird-neuroevolution/blob/main/imgs/readme/training.gif)

## To Run
Simply clone the repo and run 'python3 Game.py' in the repo and it will begin training birds. A pretrained bird is included but commented out in Game.py if you would like to skip the training process to see how one performs.

Dependencies needed:
- Python 3.6+
- NumPy
- Pygame

## What is this?
Flappy Bird (the game) but it plays itself through a neural network using a NEAT algorithm in which the neural network's hyperparameters are set through a genetic algorithm. 

The art assets and game code are taken and modified from "Tech With Tim's 'AI plays flappy bird series'." However, the neural net and evolutionary algorithm are coded from scratch with the help of NumPy for the vectorization and matrix capabilities. The game itself was coded with the help of the Pygame library.

This was done as a learning to experiment with hyperparameter tuning through a genetic algorithm and to be more comfortable with neural networks.

## Case
One idea that could have been implemented was to screen grab every frame of the game, turn it into a matrix of values based on a grayscale numerical value unrolled into a vector and fed into neural network. While this would have been interesting to implement, 5 variable parameters were fed instead.

![Image of parameters based on game](https://github.com/rxharja/flappybird-neuroevolution/blob/main/imgs/readme/Screen%20Shot%202020-10-04%20at%208.16.52%20PM.png)

- x scale distance to the next pipe
- y scale distance to the middle of the next pipe's gap
- the movement velocity of the bird as it flaps or dives
- The position of the top pipe
- The position of the bottom pipe

## The Neural Net Architecture and the Evolutionary Algorithm
![Neural Net Architecture](https://github.com/rxharja/flappybird-neuroevolution/blob/main/imgs/readme/Screen%20Shot%202020-10-04%20at%208.55.39%20PM.png)

A bit of testing to see what which architecture produced birds that flapped quickly revealed that the simplest one here was the best with one hidden layer containing five hidden nodes. 

One output node determined whether the bird flapped or not if the hypothesis function produced a value of over 0.5 or not. 

The weights were randomly generated between -1 and 1, and a sigmoid function was used to normalize the activation nodes on the network. It would be interesting to see how affective a ReLU implementation would be as it seemed that the neural network produced a competent bird, sometimes one that could not lose, within a few generations. Initially the input parameters were not normalized and competent birds would still generate, so it seems like there is a lot of room for good results with this case.

The evolutionary algorithm itself was implemented in way such that the ten best members out of every generation with a unique set of weights were used to seed the next generation. These ten members were then inserted into the next generation with no mutation or recombination to prevent a drop in fitness. The rest of the members had a set 3 percent chance of either a mutation or a recombination event.

The mutation events added a random number generated from Numpy's normal distribution random number function that ranged between -1 to 1, so it worked like adjusting weights when training a supervised machine learning model. It would have been interesting to see, however, if a completely random value in that range would be more affective in increasing fitness. In the future maybe two mutation methods could be implemented which a chance of either one being selected, and a dial could be shifted in the favor of one or the other depending on how well they are improving.

Recombination on the other hand simply switched the weights associated with each node between two members in the population. 

In the future, It would be interesting to track which mutation or recombination event was more successful in increasing fitness, and then apply an adjustment to the probability of a mutation event or a recombination event as the generations progress. This worked well in increasing fitness over time over a simple measure of how long the birds lived, particularly in the early generations. Members in the top 10 kept their score and could update it if they happen to beat it in future generations. New birds who beat that score can then take that member's place.

The selection function simply normalized this value and used it as the probability of that member in the top ten being selected to reproduce into the next generation. A weighted value was considered as well but that would have very strongly selected for the best member in the population, and I felt like diversity would have been better for interesting recombination events. Similarly to mutation and recombination events changing in frequency over generations, this could also be changed each generation depending on how many generations in we are. 

The objective function of the genetic algorithm was the value of how many pipes the birds managed to pass multiplied by the amount of frames the birds spent within 10 pixels of the center of the gap. 

## Final Thoughts
I was not expecting an extremely simple perceptron to perform so well but ultimately it performed the best out of every configuration. The genetic algorithm is not optimal yet, the learning rate felt slow.

Interesting things to try in the future would be to use the game pixels as the input instead of these five parameters to see how well it would perform. The objective function of the evolutionary algorithm would stay the same and hopefully it would select for birds which are flying through the pipes. 

Changes in the evolutionary algorithm can be made to improve its learning rate, like applying a dial to the mutation and recombination events, the selection process, and the list of top ten members. It would also be interesting to seed each population with birds who have different neural net architectures. Recombination events would require more thought, but I would say the size of the matrix for the weights on the bird with the simpler neural network would decide which indices to recombine. 
