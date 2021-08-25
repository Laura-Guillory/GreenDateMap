import logging
from datetime import datetime
import argparse
import xarray
import utils
import pandas
import numpy
import math
import multiprocessing
import os

logging.basicConfig(level=logging.WARN, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d  %H:%M:%S")
LOGGER = logging.getLogger(__name__)


def main():
    # Get command line arguments
    options = get_options()
    # Used to set the logging level from command line arguments (DEBUG, INFO, WARN, etc. Defaults to DEBUG)
    LOGGER.setLevel(options.verbose)
    # Start and end times will only show in INFO logging level or higher
    start_time = datetime.now()
    LOGGER.info('Starting time: ' + str(start_time))

    calc_green_date(options)

    end_time = datetime.now()
    LOGGER.info('End time: ' + str(end_time))
    elapsed_time = end_time - start_time
    LOGGER.info('Elapsed time: ' + str(elapsed_time))


def get_options():
    """
    Gets command line arguments and returns them.
    Options are accessed via options.verbose, etc.

    Required arguments: daily_rain, clay_content_percentage
    Optional arguments: verbose (v), start_date, end_date, period, rain_threshold, multiprocessing, title

    Run this with the -h (help) argument for more detailed information. (python main.py -h)

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose',
        help='Increase output verbosity.',
        action='store_const',
        const=logging.INFO,
        default=logging.WARN
    )
    parser.add_argument(
        '--daily_rain',
        help='The path of the netCDF data file for daily rain.',
        required=True
    )
    parser.add_argument(
        '--start_date',
        help='Set the start date. Used to restrict the analysis to a certain time period.',
        default=None
    )
    parser.add_argument(
        '--end_date',
        help='Set the end date. Used to restrict the analysis to a certain time period.',
        default=None
    )
    parser.add_argument(
        '--period',
        help='The number of days to calculate the Green Date over. Defaults to 3 days.',
        default=3,
        type=int
    )
    parser.add_argument(
        '--rain_threshold',
        help='The rainfall threshold for Green Date conditions to be considered met. Defaults to 30mm.',
        default=30,
        type=int
    )
    parser.add_argument(
        '--multiprocessing',
        help='Number of processes to use in multiprocessing. Options: single, all_but_one, all. Defaults to '
             'all_but_one.',
        choices=["single", "all_but_one", "all"],
        required=False,
        default="all_but_one",
    )
    parser.add_argument(
        '--output',
        help='The file to save the results to',
        default='results'
    )
    args = parser.parse_args()
    return args


def calc_green_date(options):
    """
    Calculates the 70th percentile of Green Dates over all years.
    Stores the result in results/green_date.nc

    :param options: Command line arguments obtained from get_options()
    :return:
    """
    # Open all files with rainfall data
    daily_rain = xarray.open_mfdataset(options.daily_rain, combine='by_coords')
    # Drop unnecessary variable
    daily_rain = daily_rain.drop_vars('crs', errors='ignore')
    # If a start date and end date are set via command line arguments, use those. Otherwise, use all data available
    start_date = options.start_date if options.start_date else daily_rain.time.values[0]
    end_date = options.end_date if options.end_date else daily_rain.time.values[-1]
    daily_rain = daily_rain.sel(time=slice(start_date, end_date))
    # Date index used to split data into years with each year beginning on the 1st of September and ending on the 31st
    # of August
    date_range = pandas.date_range(start_date, end_date)
    my_years = xarray.DataArray([t.year if (t.month < 9) else (t.year + 1) for t in daily_rain.indexes['time']],
                                dims='time', name='my_years', coords={'time': date_range})
    year_idxs = daily_rain.groupby(my_years).groups

    # Prepare jobs for multiprocessing. Each job receives 1 year of data, a copy of the options selected, and a filepath
    # where it will temporarily store its result.
    multiprocess_args = []
    processed_files = []
    for key, value in year_idxs.items():
        temp_filepath = '{folder}/{year}.{pid}.temp'.format(folder=options.output, year=key, pid=os.getpid())
        multiprocess_args.append((daily_rain.isel(time=value), options, temp_filepath))
        processed_files.append(temp_filepath)

    # Number of processes are selected via command line but default to the number of CPU cores available minus 1
    if options.multiprocessing == "single":
        number_of_worker_processes = 1
    elif options.multiprocessing == "all":
        number_of_worker_processes = multiprocessing.cpu_count()
    else:
        number_of_worker_processes = multiprocessing.cpu_count() - 1

    pool = multiprocessing.Pool(number_of_worker_processes)
    # Calls calc_green_date_for_year() with each process in the pool, using the arguments saved above
    pool.map(calc_green_date_for_year, multiprocess_args)
    pool.close()
    pool.join()

    # Open results for each year, which were temporarily stored. These contain the green dates for each year.
    green_date_per_year = xarray.open_mfdataset(processed_files, combine='by_coords')
    # Takes the green dates for each year and calculates the 70th percentile over all years.
    percentile_green_date = green_date_per_year.reduce(numpy.nanpercentile, dim='time', q=70)
    percentile_green_date = percentile_green_date.rename({'lon': 'longitude', 'lat': 'latitude'})
    output_path = '{folder}/green_date_{threshold}mm.nc'.format(folder=options.output, threshold=options.rain_threshold)
    utils.save_to_netcdf(percentile_green_date, output_path, logging_level=logging.INFO)
    green_date_per_year.close()
    # Removes all temporary files
    for process in multiprocess_args:
        temp_filepath = process[2]
        os.remove(temp_filepath)


def calc_green_date_for_year(args):
    """
    Calculates the green dates for a single year

    :param args: Tuple containing the data to use for calculation, the command line arguments (options), and the path
                 to save the results to
    :return:
    """
    # Unpack arguments
    (dataset, options, temp_filepath) = args
    # Get name of variable to use for calculations (assume first variable)
    var = list(dataset.keys())[0]

    # LOGGER.info('Daily rain:')
    # LOGGER.info(dataset['daily_rain'].values[:, 400, 400])
    # Calculate total rainfall over the period (command line argument, defaults to 3 days)
    sum_over_x_days = dataset[var].rolling(
        time=options.period,
        min_periods=1
    ).construct('window').sum('window', min_count=1).compute()
    # LOGGER.info('3 Day Sum:')
    # LOGGER.info(sum_3day.values[:, 400, 400])
    # Set up empty array to be filled with green dates
    green_dates = numpy.full(shape=(1, dataset.lat.size, dataset.lon.size), fill_value=numpy.nan, dtype=float)

    # Iterate through lats, lon
    for lat_i in range(sum_over_x_days.lat.size):
        for lon_i in range(sum_over_x_days.lon.size):
            # Iterate through time. The 1st occurrence of rainfall over the threshold (Ymm over X days, both command
            # line arguments) is the green date. If rainfall is nan, it is assumed that rain doesn't cover this region
            # and the cell is skipped. If the last day of the year is reached, the green date is assumed to be the max
            # number of days in the year. Even though the final map only shows green dates up to 1st March, green dates
            # all year round are calculated without shortcuts because they are needed to calculate an accurate
            # percentile later on.
            for time_i in range(sum_over_x_days.time.size):
                value = sum_over_x_days.values[time_i, lat_i, lon_i]
                if math.isnan(value):
                    green_dates[0, lat_i, lon_i] = numpy.nan
                    break
                if value > options.rain_threshold:
                    green_dates[0, lat_i, lon_i] = time_i
                    break
                if time_i == sum_over_x_days.time.size - 1:
                    green_dates[0, lat_i, lon_i] = sum_over_x_days.time.size
    # Insert results into a dataset to be saved
    green_dates = xarray.Dataset(
        {'green_dates': (['time', 'lat', 'lon'], green_dates)},
        coords={
            'time': [dataset.time.values[0].astype('datetime64[Y]')],
            'latitude': dataset.lat,
            'longitude': dataset.lon
        }
    )
    description = 'Green Date, which is the first date after 1 September where there is historically a 70% chance of ' \
                  'receiving at least {threshold}mm of rain over a maximum of {period} days.'\
        .format(threshold=options.rain_threshold, period=options.period)
    green_dates.green_dates.assign_attrs({'long_name': 'Green Date', 'description': description})
    utils.save_to_netcdf(green_dates, temp_filepath)


if __name__ == '__main__':
    main()
