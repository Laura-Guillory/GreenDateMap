# Green Date Map

This project contains code for calculating the expected Green Date (50mm over 3 days in 7 out of 10 years) for regions
in Northern Australia.

This software is implemented in Python and designed for handling netCDF files.

Requires daily rainfall data as input. It is recommended to obtain this from 
[LongPaddock's SILO](https://silo.longpaddock.qld.gov.au/)

## Requirements

* Python 3
* Install the packages listed in requirements.txt

## How to use

Arguments:

|||
|------------------|-----------------------------------------------------------------------------------------------------------|
|--daily_rain      |The path of the netCDF data file for daily rain.|
|--start_date      |Set the start date. Used to restrict the analysis to a certain time period.|
|--end_date        |Set the end date. Used to restrict the analysis to a certain time period.|
|--period          |The number of days to calculate the Green Date over. Defaults to 3 days.|
|--rain_threshold  |The rainfall threshold for Green Date conditions to be considered met. Defaults to 30mm.|
|--multiprocessing |Number of processes to use in multiprocessing. Options: single, all_but_one, all. Defaults to all_but_one.|
|--title           |The title of the map produced. Defaults to no title.|
|-v, --verbose     |Increase output verbosity.|

```commandline
Usage: python main.py --daily_rain PATH
```

## Contacts

**Laura Guillory**  
_Web Developer_  
Centre for Applied Climate Science  
University of Southern Queensland  
[laura.guillory@usq.edu.au](mailto:laura.guillory@usq.edu.au)