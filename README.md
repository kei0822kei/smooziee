smooziee
========

Overview
scripts/smooziee-phscat.py - plot raw data, raw data with peak spotted and raw data with smoothed


## smooziee-phscat.py
find out peak, and then fit to the data using the same number of lorentzian as peaks you found

   find out peak; use 'scipy.signal.argrelmax'
   fitting; set initial parameter by fitting each lorentzian to the data,
            and then execute grid search, whose params are shifted from inital parameter


## Usage
plot raw data  
% smooziee-phscat.py --filename="path/to/smooziee/example/KCl_GXL511_m0p25_RT_4" --run_mode='raw'

find peak  
% smooziee-phscat.py --filename="path/to/smooziee/example/KCl_GXL511_m0p25_RT_4" --run_mode='peak' --order=10

smoothing  
% smooziee-phscat.py --filename="path/to/smooziee/example/KCl_GXL511_m0p25_RT_4" --run_mode='smooth' --order=10 --param_A="2 0.02" --param_x0="3 0.5" --param_d="1 0.03"

see more infomation  
% smooziee-phscat -h


## Author

[kei0822kei](https://github.com/kei0822kei)
