from types import SimpleNamespace

import torch 
from torch_geometric.data import Data 

SUSCEPTIBLE=0; INFECTED=1; REMOVED=2
R0 = 1.5; AVG_CURE_TIME=500; INIT_INFECTED=10

def build_population(n, deg_lambd=1):
    popularity = torch.empty(n).exponential_(lambd=deg_lambd)
    cleanliness = torch.empty(n).normal_()
    healing_time = torch.empty(n).normal_().abs()
    sir = torch.zeros(n,3) # Suceptable, Recovered, Removed

    sir[:,SUSCEPTIBLE] = 1
    return SimpleNamespace(
        popularity=popularity,
        cleanliness=cleanliness,
        healing_time=healing_time,
        state=sir,
        size=n
    )

def simulate_interactions(population, interactions, r0):
    '''
    If a sick person interacts with a susceptible person, 
    and the susceptible person is not "careful" (measured by cleanliness)
    they are infected
    '''
    src,dst = interactions

    possible_src_infections = src[
        population.state[src,SUSCEPTIBLE].logical_and(
            population.state[dst,INFECTED]
        )
    ]
    possible_dst_infections = dst[
        population.state[dst,SUSCEPTIBLE].logical_and(
            population.state[src,INFECTED]
        )
    ]
    possible_infections = torch.cat(
        [possible_src_infections,possible_dst_infections]
    )
    health = population.cleanliness[possible_infections]
    infected = possible_infections[
        health <= torch.empty(
            possible_infections.size()
        ).normal_()*r0
    ]

    population.state[infected, INFECTED] = 1 
    population.state[infected, SUSCEPTIBLE] = 0


def episode(population, n_edges=100, dataset_f=None, r0=R0, cure_time=AVG_CURE_TIME, init_inf=INIT_INFECTED):
    log = []
    edges = []
    states = []

    # How long it will take each person to clear the infection
    infection_times = population.healing_time*cure_time
    
    first_iter = True 
    while population.state.sum(dim=0)[INFECTED] or first_iter:
        interactions = torch.multinomial(
            population.popularity, 
            n_edges*2, replacement=False
        ).reshape(2,n_edges)

        # Make sure first interaction involves infected person
        if first_iter:
            ground_zero = interactions[0][:init_inf]
            population.state[ground_zero, INFECTED] = 1 
            population.state[ground_zero, SUSCEPTIBLE] = 0
            first_iter = False 
        
        simulate_interactions(population,interactions, r0)
        infection_times[population.state[:,INFECTED].bool()] -= 1
        cleared = infection_times<=0 
        
        if dataset_f:
            edges.append(interactions)
            states.append(population.state.clone())

        if cleared.sum():
            population.state[cleared, INFECTED] = 0 
            population.state[cleared, REMOVED] = 1 

        log.append(population.state.sum(dim=0))

    if dataset_f:
        data = []
        static_feats = torch.stack([
            population.popularity, 
            population.cleanliness, 
            population.healing_time
        ])

        for i in range(len(edges)):
            data.append(
                Data(
                    x=static_feats, 
                    edge_index=edges[i], 
                    y=states[i]
                )
            )

        torch.save(data, dataset_f)

    return torch.stack(log)