import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import json

def find_suppliers(origin, week, current_process, raw_df):
    
    """
    For a specific demand(current_process) we identify suppliers (next processes) 
    that have a process time equal to or less than the process time (week) 
    of the current demand, and share the same destination as the origin of 
    the current process. This is done to determine the potential suppliers 
    for a specific demand (current process).

    """
    
    # extract data from the original data
    index = raw_df.loc[(raw_df['Destination']==origin) & 
                             (raw_df['Week']<=week) & 
                             (raw_df['Process']==current_process)].index.to_list()
    
    return index


def create_viable_arc_matrix(sub_current_df, supplier_idx, nested_idx):
    """
    We create a matrix to represent eligible transportation arcs (supply-demand links)
    """
    # Get the dimensions of the supply and demand
    num_demand = sub_current_df.shape[0]
    num_supply = len(supplier_idx)

    # Initialize the matrix of transportation arcs with zeros
    arcs = [[0 for _ in range(num_demand)] for _ in range(num_supply)]
    
    matches = [(val,idx_sub) for idx_sub,sublist in enumerate(nested_idx) 
               for val in sublist]

    for val in matches:
        n = supplier_idx.index(val[0])
        arcs[n][val[1]] = 1
        
    return arcs


def solve_supply_demand_balancing(supply_df, demand_df, arcs):
    """   
    We solve a Linear Program model to allocate resources between the supply 
    and demand points. The viable supply-demand links are represented 
    by arcs matrix. Note that the objective function is set to zero.
    """
    
    supply = list(supply_df['Amount'])
    demand = list(demand_df.loc[:,'Amount'])

    # defining sets of supply and demand
    I = np.arange(len(supply))
    J = np.arange(len(demand))

    # Model set up
    m = gp.Model('Supply and Demand Balancing')

    # variable x: Amount of demand of j that is satisfied by supplier i
    x=m.addVars(I,J,vtype=GRB.CONTINUOUS, name="x")

    # supply constraint
    for i in I:
        m.addConstr(sum(arcs[i][j] * x[i,j] for j in J) == supply[i])

    # demand constraint
    for j in J:
        m.addConstr(sum(arcs[i][j] * x[i,j] for i in I) == demand[j])

    # objective function is set to zero
    m.setObjective( 0 , GRB.MINIMIZE)
    
    # Set the OutputFlag parameter to 0 to suppress solver output
    m.setParam(GRB.Param.OutputFlag, 0)

    m.optimize()

    # Check if the optimization was successful
    if m.status == GRB.OPTIMAL:
        # Get the solution
        sol = np.array([var.x for var in m.getVars()])
    else:
        print("No feasible solution found.")

    # Dispose of the model to free up resources
    m.dispose()
    
    return sol.reshape(len(I),len(J))


def process_sub_solution(solution, demand_idx, supplier_idx, potential_df):
    """
    We process the solution of the subproblem and
    match it with its next and current process data
    """
    # Count non-zero values along each column of solution
    num = np.count_nonzero(solution, axis=0)

    # Creating dataframe for next process based on the balancing solution
    sub_next_df = pd.DataFrame(data = np.zeros([sum(num), len(col_selection)])*np.nan,
                                   columns = [col_selection])

    k = 0
    for j in range(len(demand_idx)):
        for i,val in enumerate(supplier_idx):
            if solution[i][j] > 0:
                sub_next_df.iloc[k,:] = potential_df.loc[val,:]
                sub_next_df.loc[k,'Amount'] = solution[i][j]
                k += 1

    new_index = []
    for idx,count in enumerate(num):
        new_index += list(sub_current_df.index[idx]+np.linspace(0, 1, count+1)[:-1])

    sub_next_df.index = new_index
    
    return sub_next_df


if __name__ == "__main__":

    # Input data
    with open('conf.json', 'r') as f:
        conf = json.load(f)

    raw_data = pd.read_csv(conf['input_csv'])

    maping_columns = {'for_process':'Process', 'to_processing_cnt':'Destination', 'send_from_cnt':'Origin'}
    raw_data = raw_data.rename(columns=maping_columns)

    # initial demand (process = Delivery)
    demand_data = raw_data[raw_data['Process']=='Delivery']
    demand_data = demand_data.reset_index(drop=True)
    demand_data['Demand'] = demand_data.index + 1

    # Backward list of processes
    process_lst = list(raw_data['Process'].unique()[::-1])
    process_lst.remove('Delivery')

    # Initializing the output dataframe 
    col_selection = ['Origin','Process', 'Destination', 'Week', 'Amount']
    output_data = demand_data[col_selection+['Demand']]

    # Iterative Algorithm
    for process_idx, process in enumerate(process_lst):
        
        process = process_lst[process_idx]
        current_process_df = output_data.iloc[:,0:5]
        next_process_df = pd.DataFrame()

        # Defining unique origins in the current process
        origin_lst = list(set(current_process_df['Origin']))
        origin_lst
        
        # Loop over to solve subprblems (current processes with same origin)
        for origin in origin_lst:
            
            # extract all current processes with the specific origin
            sub_current_df = current_process_df[current_process_df['Origin'] == origin]

            # identify all potential next processes for the sub_problem (specific origin)
            nested_idx = []
            for idx in sub_current_df.index:
                week = sub_current_df.loc[idx, 'Week']
                nested_idx.append(find_suppliers(origin, week, process, raw_data))

            supplier_idx = sorted(list(set(val for sublist in nested_idx for val in sublist))) # unique suppliers

            # extract data of potential and eligible suppliers
            potential_df = raw_data.loc[supplier_idx,col_selection]

            # define viable transportation arcs
            arcs = create_viable_arc_matrix(sub_current_df, supplier_idx, nested_idx)

            # solve supply and demand balancing problem
            solution = solve_supply_demand_balancing(potential_df, sub_current_df, arcs)

            # create dataframe of the next processes
            demand_idx = sub_current_df.index
            sub_next_df = process_sub_solution(solution, demand_idx, supplier_idx, potential_df)

            # integrate the next process dataframes of subproblems
            next_process_df = pd.concat([sub_next_df,next_process_df],axis=0)
            
            
        # integrating next process dataframe with the output dataframe
        output_data = pd.concat([next_process_df, output_data], axis=1).sort_index()
        output_data.columns = col_selection*(process_idx+2) + ['Demand']
        output_data = output_data.reset_index(drop=True)
    
    # modifying output data
    output_data = output_data.drop(columns=['Origin'])
    output_data.rename(columns={'Destination': 'Cnt'}, inplace=True)

    # modifying demand number
    for idx in output_data.index:
        if np.isnan(output_data.loc[idx,'Demand']):
            output_data.loc[idx,'Demand'] = output_data.loc[idx-1,'Demand']
    
    output_data.to_csv(conf['output_csv'])