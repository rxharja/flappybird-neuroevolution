import numpy as np
from NeuralNet import NeuralNet
from Bird import Bird


class Population:

    def __init__(self, inp, pop_size=150):
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
            nn = NeuralNet(1, 4, starting_in, 1)
            self.members.append(Bird(230, starting_pos, nn))
            # self.members.append(nn)

    def reseed(self, starting_in):
        self.members = self.select()
        for i in range(len(self.members)):
            self.members[i].y = 200 + i

        for i in range(10, len(self.members)):
            self.mutate(self.members[i])
        
        for i in range(10, len(self.members) - 1):
            self.crossover(self.members[i], self.members[i + 1])

    def select(self):
        new_pop = []
        best = sorted(self.previous, key=lambda bird: bird.alive, reverse=True)
        total = 0

        for bird in best:
            total += (bird.alive * bird.pipes_passed)

        for itm in range(len(self.best_ten)):
            for bird in best:
                if not self.best_ten[itm]:
                    self.best_ten[itm] = bird

                if self.best_ten[itm].pipes_passed < bird.pipes_passed:
                    self.best_ten[itm] = bird
                    break

        [new_pop.append(Bird(230, 0, bird.nn)) for bird in self.best_ten]

        if total > 0:
            weights = [(bird.alive * bird.pipes_passed)/total for bird in best]
            for i in range(self.pop_size - 10):
                index = 0
                r = np.random.random()
                while r > 0:
                    r = r - weights[index]
                    index += 1
                new_pop.append(Bird(230, 0, best[index].nn))
        else:
            for i in range(10, self.pop_size):
                new_pop.append(Bird(230, 0, best[i].nn))

        return new_pop

    def mutate(self, member):
        for m in member.nn.theta:
            for n in m:     
                if np.random.random() > 0.9:
                    n = 2 * np.random.random() - 1
        
    def crossover(self, member1, member2):
        for n in range(len(member1.nn.theta[0])):
            if np.random.random() > 0.9:
                temp1 = member1.nn.theta[:,n]
                member1.nn.theta[:,n] = member2.nn.theta[:,n]
                member2.nn.theta[:,n] = temp1
