## Script version from accelerate notebook

Creating a full python script from the accelerate_notebook

-   [x] multigpu by using accelerate
-   [x] saving a loading models
-   [x] Allow to skip the data preparation step by using a precomputed object (Lets share the gdrive link here or upload the file)
-   [x] add the additional metrics (the slow kl divergence metric calculated per cell type)
-   [ ] Provide the config file to run using accelerate
-   [ ] Provide the enviroment to run the scripts (conda env)
-   [x] Fix the weights and bias metrics (Should we use this per user or have some credentials for the project?)
-   [ ] generalize the script for more cell types
-   [ ] Work in the paralelization issues (sampling using all the GPUS instead wait for the accelerate main process finish)
    -   This may not be possible due to gimmemotif caching requirements. Still researching how to resolve this
