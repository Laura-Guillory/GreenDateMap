import logging
from datetime import datetime
import argparse
import xarray
import utils
import pandas
import numpy


logging.basicConfig(level=logging.WARN, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d  %H:%M:%S")
LOGGER = logging.getLogger(__name__)


def main():
    options = get_options()
    LOGGER.setLevel(options.verbose)
    start_time = datetime.now()
    LOGGER.info('Starting time: ' + str(start_time))

    green_date = calc_green_date(options)
    utils.save_to_netcdf(green_date, 'test.nc')

    end_time = datetime.now()
    LOGGER.info('End time: ' + str(end_time))
    elapsed_time = end_time - start_time
    LOGGER.info('Elapsed time: ' + str(elapsed_time))


def get_options():
    """
    Gets command line arguments and returns them.
    Options are accessed via options.verbose, etc.

    Required arguments: daily_rain
    Optional arguments: verbose (v)

    Run this with the -h (help) argument for more detailed information. (python main.py -h)

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose',
        help='Increase output verbosity',
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
    args = parser.parse_args()
    return args


def calc_green_date(options):
    daily_rain = xarray.open_mfdataset(options.daily_rain)
    daily_rain = daily_rain.drop_vars('crs', errors='ignore')
    start_date = options.start_date if options.start_date else daily_rain.time.values[0]
    end_date = options.end_date if options.end_date else daily_rain.time.values[-1]
    date_range = pandas.date_range(start_date, end_date)
    daily_rain = daily_rain.sel(time=slice(start_date, end_date))

    my_years = xarray.DataArray([t.year if (t.month < 9) else (t.year + 1) for t in daily_rain.indexes['time']],
                                dims='time', name='my_years', coords={'time': date_range})
    daily_rain = daily_rain.groupby(my_years).apply(calc_green_date_for_year)
    return daily_rain


def calc_green_date_for_year(dataset: xarray.Dataset):
    var = list(dataset.keys())[0]
    dataset[var] = dataset['daily_rain'].rolling(time=3, min_periods=3).construct('window').sum('window')
    return dataset


def create_map():
    pass


if __name__ == '__main__':
    main()
