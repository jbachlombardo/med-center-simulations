# med-center-simulations

Med center process flow simulations for ISP:

- 1_ (28/4)
  - Queueing model using normal distributions and estimated times by math
  - Simulation model using normal distributions, hourly approach
- 2_ (29/4)
  - Simulation model using skewed distributions, manually moved forward on minute basis to understand period-by-period patient movement
  - Simulation model using skewed distributions, looped over a single step
- 3_ (30/4)
  - Simulation model using skewed distributions, looped over multiple steps in notebook -- single simulation with visualisation
  - Simulation model using skewed distributions, looped over multiple steps in terminal -- n_simulations with mean results from simulations
- 4_ (2/5)
  - Simulation model exploring different arrival distributions eg bunched poisson and normal arrivals
  - **Troubleshooting (10/5)**: Fixed issue with wait list to service, see this for documentation; fixed in 4_ update and 6_ update
- 5_ (3/5) 
  - Simulation model incorporating random wait times of provider (provider not available immediately when patient is arrives)
  - Terminal version provides output to csv for summarizing and analyzing
- 6_ (10/5)
  - Testing variations
- 7_ (10/5)
  - Simulation variations (terminal)
- 8_ (19/5)
  - Adding follow up step, which required rewriting several functions to accomodate shared servers between refine_complaint and followup steps
