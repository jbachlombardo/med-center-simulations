# med-center-simulations

Med center process flow simulations for ISP:

- 11_ (12/6)
  - Rewrite to change operating model of patient flow such that:
    - Single `flow_staff` tied to patient throughout the `refine_complaint`, `exam`, and `follow_up` steps
    - Incorporating `offload` to simulate `providers` offloading a percentage of their work onto the tied `flow_staff`
- 10_ (28/5)
  - Rewriting code to increase simulation speed. ~15x speed increase achieved (~7.4s / sim to ~0.5s / sim) by generating time distributions for each simulation at the beginning of the simulation and then randomly drawing from these distributions within each period of the simulation, rather than generating a new distribution within each period of the simulation.
    - Jupyter `rewrite_sample_testing` to confirm that the change in distribution generation approach does not alter the distributions being drawn from such that service times in the simulations are different. T-tests fail to reject that the samples are significantly different.
- 9_ (20/5)
  - Adding lunch break and ending hours, which compresses arrivals
  - **Fix (24/5)**: Fixed leakage in adding to wait list due to sorting wait list dict keys at 10 causing overwriting of patients in wait queue
- 8_ (19/5)
  - Adding follow up step, which required rewriting several functions to accomodate shared servers between refine_complaint and followup steps
- 7_ (10/5)
  - Simulation variations (terminal)
- 6_ (10/5)
  - Testing variations
- 5_ (3/5) 
  - Simulation model incorporating random wait times of provider (provider not available immediately when patient is arrives)
  - Terminal version provides output to csv for summarizing and analyzing
- 4_ (2/5)
  - Simulation model exploring different arrival distributions eg bunched poisson and normal arrivals
  - **Fix (10/5)**: Fixed issue with wait list to service, see this for documentation; fixed in 4_ update and 6_ update
- 3_ (30/4)
  - Simulation model using skewed distributions, looped over multiple steps in notebook -- single simulation with visualisation
  - Simulation model using skewed distributions, looped over multiple steps in terminal -- n_simulations with mean results from simulations
- 2_ (29/4)
  - Simulation model using skewed distributions, manually moved forward on minute basis to understand period-by-period patient movement
  - Simulation model using skewed distributions, looped over a single step
- 1_ (28/4)
  - Queueing model using normal distributions and estimated times by math
  - Simulation model using normal distributions, hourly approach
