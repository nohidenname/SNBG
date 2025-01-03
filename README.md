# SNBG
SNBG provides a group-based brain network construction method to build a single-sample brain network on resting-state functional magnetic resonance imaging data of Alzheimer's disease, aiming to reduce the intra-group differences of samples. The developer is Yikun Zhou from Laboratory of Biomedical Network,Xiamen University of China.

# Overview of SNBG
![6bd0eab7f7096e2eb80385e6602fa3b9.png](https://ice.frostsky.com/2025/01/02/6bd0eab7f7096e2eb80385e6602fa3b9.png)
The process for calculating sample networks using the SNBG method.

# Requirement

    python == 3.10.6    
    libsvm == 3.23.0.4
    networksx == 3.1
    numpy == 1.22.3
    matplotlib == 3.6.2
    scikit-learn == 1.0.2


# Quick start
#  Input
The input is a processed nii image, the acquisition time point is 130, and the brain atlas is 268 brain regions
# Run SBNG method
If you want to get the time series, run: get_time_series_data.ipynb

If you want to use the SNBG method, use your own time series dataset as input, run: snbg.ipynb

If you want to use the pcc-based method, use your own time series dataset as input, run:  pcc_based.ipynb

If you want to calculate the RSD, use your own time series dataset as input, run: rsd_calculate.ipynb

All functions are defined in functions.py

# Contact
Please contact me if you have any help: zhouyikun@stu.xmu.edu.cn
