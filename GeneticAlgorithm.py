import numpy as np
from NeuralNet import NeuralNet
from Bird import Bird
import copy


class Population:

    def __init__(self, inp, pop_size=75):
        self.generation = 1
        self.members = []
        self.previous = []
        self.best_ten = [None] * 10
        self.pop_size = pop_size
        self.seed(inp)
        
    def seed(self, starting_in):
        for i in range(0, self.pop_size):
            starting_pos = 100 + i
            starting_in[0] = starting_pos
            nn = NeuralNet(1, 4, starting_in)
            self.members.append(Bird(230, starting_pos, nn))

    def reseed(self, starting_in):
        self.get_best()
        # print([bird.pipes_passed > 1 for bird in self.best_ten])
        if not np.any([bird.pipes_passed > 1 for bird in self.best_ten]):
            print("resetting population from scratch")
            self.generation = 0
            self.best_ten = [None] * 10
            self.seed(starting_in)
        else:
            print("reseeding population with best")
            self.members = self.select()
            for i in range(len(self.members)):
                self.members[i].y = 200 + i
                if self.members[i].mutate:
                    self.members[i].nn.theta = self.mutate(self.members[i])

            for i in range(len(self.members) - 1):
                if self.members[i].mutate and self.members[i + 1].mutate:
                    self.members[i].nn.theta, self.members[i + 1].nn.theta = \
                        self.crossover(self.members[i], self.members[i + 1])

    def get_best(self):
        best = sorted(self.previous, key=lambda bird: bird.alive * bird.pipes_passed, reverse=True)
        l = len(self.best_ten)
        for bird in best:
            bird_weights = bird.nn.theta
            for itm in range(l):
                i = l - itm - 1
                if not self.best_ten[i]:
                    self.best_ten[i] = bird
                    break
                else:
                    best_bool = [np.any(best.nn.theta != bird_weights) for best in self.best_ten]
                    if (self.best_ten[i].alive < bird.alive and self.best_ten[i].pipes_passed < bird.pipes_passed):
                        if np.all(best_bool):
                            print("new member in best")
                            self.best_ten[i] = bird
                        else:
                            if np.any(self.best_ten[i].theta == bird_weights):
                                self.best_ten[i].alive = bird.alive
                                self.best_ten[i].pipes_passed = bird.pipes_passed
                        break

        self.best_ten.sort(key=lambda bird: bird.alive * bird.pipes_passed, reverse=True)
        print([bird.pipes_passed*bird.alive for bird in self.best_ten])
        with open("best_each_gen.txt", "a") as f:
            f.write(str(self.generation) + "\t" + str(self.best_ten[0].nn.theta) + "\n")

    def select(self, weight=0.99):
        new_pop = []
        [new_pop.append(Bird(230, 0, copy.deepcopy(bird.nn), False)) for bird in self.best_ten]

        weights = [weight]
        for itm in self.best_ten:
            weight **= 2
            weights.append(weight)

        for i in range(self.pop_size - 10):
            index = 0
            r = np.random.random()
            while r > 0:
                r = r - weights[index]
                index += 1
                if index > len(self.best_ten) - 1:
                    index = 0
            new_pop.append(Bird(230, 0, copy.deepcopy(self.best_ten[index].nn), True))

        return sorted(new_pop, key=lambda bird: bird.mutate)

    def mutate(self, member):
        for m in range(len(member.nn.theta)):
            for n in range(len(member.nn.theta[m])):     
                if np.random.random() > 0.97:
                    member.nn.theta[m,n] = 2 * np.random.random() - 1
        return np.copy(member.nn.theta)
        
    def crossover(self, member1, member2):
        for n in range(len(member1.nn.theta[0])):
            if np.random.random() > 0.97:
                temp1 = member1.nn.theta[:,n]
                member1.nn.theta[:,n] = member2.nn.theta[:,n]
                member2.nn.theta[:,n] = temp1
        return np.copy(member1.nn.theta), np.copy(member2.nn.theta)
