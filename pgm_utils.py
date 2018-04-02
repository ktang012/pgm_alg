import math, itertools, copy
import numpy as np

from pgmpy.factors.discrete import TabularCPD, DiscreteFactor

def get_gibbs_scope(factors):
    ''' Input: list of factors
        Get all of the variables defined by the factors '''
        
    variables = set()
    for f in factors:
        fscope = set(f.scope())
        variables = variables.union(fscope)
    return variables
    
def init_Q(factors, seed=1):
    ''' Input: list of factors
        Initialize a uniform distribution over the variables 
        defined by the factors '''
    np.random.seed(seed=1)
    Q = {}
    for f in factors:
        factor_cardinality = f.get_cardinality(f.scope())
        for var,card in factor_cardinality.items():
            if var not in Q:
                # cpd_values = [1/card for i in range(card)]
                # Q[var] = TabularCPD(var, card, [cpd_values])   
                Q[var] = DiscreteFactor(variables=[var], cardinality=[card],
                                        values=[np.random.uniform(1,100,size=(card))])
                Q[var].normalize()
    return Q
  
def get_assignment_index(factor, assignment):
    ''' Input: factor, dictionary assignment
        Get the indices associated with some assignment 
        for a factor. If a variable (i.e. key) is missing, 
        it will return a slice of all assignments associated 
        with that variable '''
        
    indices = []
    for var in factor.scope():
        if var in assignment:
            indices.append(assignment[var])
        else:
            indices.append(slice(factor.get_cardinality([var])[var]))

    for bound in range(factor.values.ndim):
        if factor.values.shape[bound] <= indices[bound]:
            indices[bound] = factor.values.shape[bound] - 1
    indices = tuple(indices)
    
    return indices
    
def get_numeric_cardinality(factor,exclude=None):
    ''' Input: factor, list of variables to exclude
        Get the numeric cardinality of a cpd
        subject to variables to exclude '''
        
    if exclude != None:
        fscope = list(factor.scope())
        for var in exclude:
            fscope.var(exclude)
        card = factor.get_cardinality(fscope)
    else:
        card = factor.get_cardinality(factor.scope())
    
    num_cardinality = 1
    for i in card:
        num_cardinality *= card[i]
    return num_cardinality
    
def get_factors_with_variable(factors, variable):
    ''' Input: list of factors, variable
        Get all factors with variable in its scope '''
        
    factors_with_variable = set()
    for f in factors:
        if variable in f.scope():
            factors_with_variable.add(f)
    return factors_with_variable
    
def get_all_assignments(cardinality):
    ''' Input: dictionary of variables to domains
        Get all combinations of assignments given
        variables and domains '''
    
    variables = []
    domains = []
    all_assignments = []
    
    for var in cardinality:
        domains.append(range(cardinality[var]))
        variables.append(var)
    
    for vals in itertools.product(*domains):
        assignment = {}
        for var, val in zip(variables, vals):
            assignment[var] = val
        all_assignments.append(assignment)
    
    return all_assignments
    
def marginalize_multivariate_factors(factors):
    ''' Marginalizes multivariate factors to get
        univariate factors for Bethe cluster graph
        construction '''
    
    univariates = set()
    marginalized_factors = []
    for f in factors:
        if len(f.scope()) > 1:
            fscope = set(f.scope())
            fvar = set(f.variable)
            fdiff = fscope.difference(fvar)
            marg_f = f.marginalize(fdiff, inplace=False)
            marginalized_factors.append(marg_f)
            univariates.add(f.variable)
        else:
            univariates.add(f.variable)
    return marginalized_factors
    
def discretize_factors(factors):
    ''' Turns CPDs to discrete factors '''
    discretized_factors = []
    for f in factors:
        discretized_factors.append(f.to_factor())
    return discretized_factors
    
    
    
    
    
    
    
    
    
    
    
    
    
    
