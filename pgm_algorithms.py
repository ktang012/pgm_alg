import math, itertools, copy
import numpy as np

from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.models.ClusterGraph import ClusterGraph

from pgm_utils import *

class MeanFieldInference:
    ''' Algorithm 11.7 '''
    def __init__(self, factors, Q=None, seed=1):
        if Q == None:
            self.Q = init_Q(factors, seed)
        else:
            self.Q = Q
            
        self.factors = []
        for f in factors:
            self.factors.append(f.copy())
            
    def get_marg_products(self, assignments):
        product = 1
        for var, assn in assignments.items():
            product *= self.Q[var].values[assn]
        return product
        
    def fixed_point_optimize(self, x_i, relevant_factors):
        new_x_i = 0
        for f in relevant_factors:
            U_phi_cardinality = dict(f.get_cardinality(f.scope()))
            del U_phi_cardinality[x_i[0]]
            U_phi_assignments = get_all_assignments(U_phi_cardinality)
            
            for u_phi in U_phi_assignments:
                marg_product = self.get_marg_products(u_phi)
                u_phi[x_i[0]] = x_i[1]
                index = get_assignment_index(f, u_phi)
                new_x_i += marg_product * f.values[index]
                
        return math.exp(new_x_i)
        
    def mean_field_approximation(self):
        unprocessed = get_gibbs_scope(self.factors)
        count_iterations = 0
        processed_history = []
        while len(unprocessed) != 0:
            count_iterations += 1
            processed_history.append(len(unprocessed))
            variable_to_optimize = unprocessed.pop()
            old_X_i = self.Q[variable_to_optimize].copy()
            num_cardinality = get_numeric_cardinality(self.Q[variable_to_optimize])
            relevant_factors = get_factors_with_variable(self.factors, variable_to_optimize)

            for i in range(num_cardinality):
                variable_assignment = (variable_to_optimize, i)    
                index = get_assignment_index(self.Q[variable_to_optimize], {variable_to_optimize: i})
                self.Q[variable_to_optimize].values[index] = self.fixed_point_optimize(variable_assignment,
                                                                                       relevant_factors)
                                                                             
            self.Q[variable_to_optimize].normalize(inplace=True)
            
            if not np.allclose(self.Q[variable_to_optimize].values, old_X_i.values):
                vars_to_process = get_gibbs_scope(relevant_factors)
                unprocessed = unprocessed.union(vars_to_process)
                
            if variable_to_optimize in unprocessed:
                unprocessed.remove(variable_to_optimize)
        history = {"processed": processed_history,
                   "iterations": count_iterations}
        return history 
         
class LoopyBeliefPropagation:
    ''' Loopy Belief Propagation on Bethe Cluster graphs, using
        belief message update scheme with dampenining '''
    def __init__(self, factors):
        self.factors = []
        for f in factors:
            self.factors.append(f.copy())
           
    def __create_bethe_cluster_graph(self):
        bethe_cg = ClusterGraph()
        uni = []
        multi = []
        for f in self.factors:
            if len(f.scope()) == 1:
                uni.append(f.scope())
            else:
                multi.append(f.scope())
        
        for mf in multi:
            for uf in uni:
                if uf[0] in mf:
                    bethe_cg.add_edge(uf[0], tuple(mf))
                    
        bethe_cg.add_factors(*self.factors)
        
        self.cluster_graph = bethe_cg
        
    def __init_cluster_graph(self):
        self.__create_bethe_cluster_graph()
        self.cluster_beliefs = {}
        self.sepset_beliefs = {}
    
        for node in self.cluster_graph.nodes():
            init_belief = False
            for factor in [self.cluster_graph.get_factors(node)]:
                if init_belief == False:
                    init_belief = factor.copy()
                else:
                    init_belief.product(factor, inplace=True)
            self.cluster_beliefs[node] = init_belief.copy()
        
        for edge in self.cluster_graph.edges():
            if len(edge[0]) < len(edge[1]):
                self.sepset_beliefs[edge] = self.cluster_graph.get_factors(edge[0]).identity_factor()
            else:
                self.sepset_beliefs[edge] = self.cluster_graph.get_factors(edge[1]).identity_factor()
        
    def loopy_belief_propagation(self, lambd=1, num_iterations=100):
        self.__init_cluster_graph()
        
        reverse_pass = {}
        for edge in self.cluster_graph.edges():
            reverse_pass[edge] = True
        
        count_iterations = 0
        curr_it = 0
        message_convergence_history = []
        while True:
            has_converged = True
            curr_it += 1
            messages_not_converged = 0
            count_iterations += 1
            for edge in self.cluster_graph.edges():
                node_i = edge[0]
                node_j = edge[1]

                scope_i = set(node_i)
                scope_j = set(node_j)
                if len(scope_i) == 1:
                    sepset_scope = scope_i
                else:
                    sepset_scope = scope_j
                diff_scope_i = scope_i.difference(sepset_scope)
                diff_scope_j = scope_j.difference(sepset_scope)
                
                marg_belief_i = self.cluster_beliefs[node_i].marginalize(diff_scope_i, inplace=False)
                marg_belief_j = self.cluster_beliefs[node_j].marginalize(diff_scope_j, inplace=False)
                
                if not np.allclose(marg_belief_i.normalize(inplace=False).values, 
                                   marg_belief_j.normalize(inplace=False).values):
                    if curr_it == num_iterations:
                        diff = np.abs(marg_belief_i.normalize(inplace=False).values - 
                                      marg_belief_j.normalize(inplace=False).values)
                        # print(diff)
                    
                    messages_not_converged += 1
                    has_converged = False
                    if reverse_pass[edge]:
                        prev_cbelief = self.cluster_beliefs[node_j].copy()
                        self.cluster_beliefs[node_j] *= (marg_belief_i / self.sepset_beliefs[edge]) * lambd
                        self.cluster_beliefs[node_j] += prev_cbelief * (1 - lambd)
                        
                        prev_sbeliefs = self.sepset_beliefs[edge].copy()
                        self.sepset_beliefs[edge] = (marg_belief_i.copy() * lambd) + (1-lambd) * prev_sbeliefs
                        reverse_pass[edge] = False
                    else:
                        prev_cbelief = self.cluster_beliefs[node_i].copy()
                        self.cluster_beliefs[node_i] *= (marg_belief_j / self.sepset_beliefs[edge]) * lambd
                        self.cluster_beliefs[node_i] += prev_cbelief * (1 - lambd)
                        
                        prev_sbeliefs = self.sepset_beliefs[edge].copy()
                        self.sepset_beliefs[edge] = (marg_belief_j.copy() * lambd) + (1-lambd) * prev_sbeliefs
                        reverse_pass[edge] = True
                else:
                    print(node_i, node_j, "has converged at iteration", curr_it)
                
            message_convergence_history.append(messages_not_converged)    
            if curr_it >= num_iterations:
                print("Loopy BP has timed out.")
                break
            elif has_converged:
                print("Loopy BP has converged.")
                break
                
        history = {"convergence": message_convergence_history,
                   "iterations": count_iterations}
                   
        return history          
