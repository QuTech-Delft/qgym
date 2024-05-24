"""Metrics to assess the performance of Routers.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

import networkx as nx
import numpy as np
from numpy.typing import NDArray
import warnings
from statistics import mean
from collections import deque

class RoutingSolutionQuality:

    def __init__(
        self,
    ) -> None:
        # pylint: disable=line-too-long
        """Init of the :class:`~qgym.benchmarks.metrics.routing.RoutingSolutionQuality` class.

        """

    def interaction_gates_ratio_loss(
        self,
        initial_interaction_circuit : NDArray[np.int_] = None,
        swaps_added : int = None ,
        final_interaction_circuit : NDArray[np.int_] = None,
    ):
        """Method to calculate the ratio of the final number of interactions divided by
        the initial number of interactions.
        At least two of three need to be specified.

        Args:
            swaps_added: int, represents the number of swaps added during the routing
            process.
            initial_interaction_circuit: An array of 2-tuples of integers, where every 
            tuple represents a, not specified, gate acting on the two qubits labeled by 
            the integers in the tuples. This is the interaction circuit considered 
            before adding the swaps.
            final_interaction_circuit: An array of 2-tuples of integers, where every 
            tuple represents a, not specified, gate acting on the two qubits labeled by 
            the integers in the tuples. This is the interaction circuit considered 
            after adding the swaps.
        """
        if (initial_interaction_circuit==None) + (final_interaction_circuit==None) + (
            swaps_added==None) <2:
            warnings.warn('At least two of the input variables have to be given')
            return
        
        if final_interaction_circuit is None:
            num_initial_interactions = len(initial_interaction_circuit)
            return (num_initial_interactions+swaps_added) / num_initial_interactions
        elif initial_interaction_circuit is None:
            num_final_interactions = len(final_interaction_circuit)
            return num_final_interactions / (num_final_interactions-swaps_added)
        elif swaps_added is None:
            num_final_interactions = len(final_interaction_circuit)
            num_initial_interactions = len(initial_interaction_circuit)
            return num_final_interactions / initial_interaction_circuit
    
    def gates_ratio_loss(
        self,
        initial_number_of_gates: int = None,
        swaps_added: int = None,
        final_number_of_gates: int = None,
    ):
        """Method to calculate the ratio of the final number of gates divided by
        the initial number of gates in a circuit.

        Args:
            initial_number_of_gates: int
            swaps_added: int, represents the number of swaps added during the routing
            process.
            final_number_of_gates: int
        """
        if (initial_number_of_gates==None) + (final_number_of_gates==None) + (
            swaps_added==None) <2:
            warnings.warn('At least two of the input variables have to be given')
            return
        
        if final_number_of_gates is None:
            return (initial_number_of_gates + swaps_added) / initial_number_of_gates
        elif initial_number_of_gates is None:
            return final_number_of_gates / (final_number_of_gates - swaps_added)
        elif swaps_added is None:
            return final_number_of_gates / initial_number_of_gates

    def fidelity_ratio_loss(self,
                            initial_interaction_circuit: NDArray[np.int_],
                            connection_graph: nx.Graph,
                            swaps_gates_inserted: deque[tuple[int, int, int]]):
        """Method to calculate the ratio of the amount of fidelity lost  

        Args:
            initial_interaction_circuit: An array of 2-tuples of integers, where every 
                tuple represents a, not specified, gate acting on the two qubits labeled 
                by the integers in the tuples.
            connection_graph: ``networkx`` graph representation of the QPU topology. 
                Each node represents a physical qubit and each edge represents a
                connection in the QPU topology.
            swap_gates_inserted:  A deque of 3-tuples of integers, to register which 
                gates to insert and where. Every tuple (g, q1, q2) represents the 
                insertion of a SWAP-gate acting on logical qubits q1 and q2 before gate 
                g in the interaction_circuit.
        """
        
        #TODO: method needs thorough revision, these choices are not effident nor correct perse.
        interactions = []
        interaction_fidelities =[]
        
        for interaction in initial_interaction_circuit:
            (qubit1, qubit2) = interaction
            if interaction in interactions:
                index = interactions.index(interaction)
                interaction_fidelities[index] *= connection_graph[qubit1][qubit2]['weight']
            else:
                interactions.append(interaction)
                interaction_fidelities.append(connection_graph[qubit1][qubit2]['weight'])
        
        initial_fidelity_mean = mean(interaction_fidelities)
        
        for interaction in swaps_gates_inserted:
            (_, qubit1, qubit2) = interaction
            if interaction in interactions:
                index = interactions.index(interaction)
                interaction_fidelities[index] *= connection_graph[qubit1][qubit2]['weight']
            else:
                interactions.append(interaction)
                interaction_fidelities.append(connection_graph[qubit1][qubit2]['weight'])
        
        final_fidelity_mean = mean(interaction_fidelities)
        
        return initial_fidelity_mean / final_fidelity_mean