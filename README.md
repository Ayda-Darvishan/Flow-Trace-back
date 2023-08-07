# Flow Trace-back ALgorithm

## Algorithm Overview
The flow trace-back algorithm is designed to sort out orders from network flow results, specifically for each combination of product and treatment. The code has been developed in Python and is designed to trace back and obtain all the information for each demand through different processing steps. The Flow Trace-back Algorithm employs an iterative approach, commencing from the last process and iteratively analyzing each process until reaching the first one. At each process step, the trace back problem is viewed as a supply-demand balancing challenge, aiming to connect viable suppliers to demand and allocate supply amounts to fulfill the corresponding demand amounts.

The algorithm partitions the demand balancing assignment within each process step into subproblems in order to enhance computational efficiency. Each subproblem exclusively encompasses demand points with the same 'send_from_cnt' location, allowing for targeted optimization. The subproblems are efficiently solved using the Gurobi Optimizer, providing accurate results.

## Prerequisites
Before running the code, make sure you have the following installed:

1. Python (version 3.x recommended)
2. Gurobi Optimizer

## Gurobi Installation Instructions
1. Sign up for a Gurobi account: 
   - Go to the Gurobi website (https://www.gurobi.com/) and sign up for a free academic license or purchase a commercial license if needed.
   - Download the Gurobi Optimizer package suitable for your operating system.

2. Install Gurobi Optimizer:
   - Follow the installation instructions provided by Gurobi for your specific operating system.

3. Set up Gurobi license (if necessary):
   - If you are using a free academic license, Gurobi should automatically pick it up during the installation.
   - If you have a commercial license, you might need to set up the license environment variables or file as instructed in the Gurobi documentation.


## Running the Code
1. Clone or download the code from https://github.com/Ayda-Darvishan/flow-trace-back-algorithm.git.
2. Navigate to the project directory in the terminal or command prompt.
3. Before running the code, make sure you have set up Gurobi as instructed in the "Gurobi Installation Instructions" section.
4. Define the path to the input and output data in `conf.json`, which is located in the same directory as the Python script.
5. Execute the Python script `flow_trace_back.py`
