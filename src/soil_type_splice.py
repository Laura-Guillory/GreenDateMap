import logging
from datetime import datetime
import argparse
import xarray
import utils
import numpy

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

    regrid_clay_content(options)
    splice_maps(options)

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
        '--clay_content',
        help='The path of the netCDF data file for soil clay content (percentage).',
        required=True
    )
    parser.add_argument(
        '--green_date_files',
        help='The path of the netCDF data file for Green Date results',
        default='results/green_date_{threshold}mm.nc'
    )
    parser.add_argument(
        '--output',
        help='The path to save the combined Green Date results at.',
        default='results/green_date_soil.nc'
    )
    args = parser.parse_args()
    return args


def regrid_clay_content(options):
    clay_content = xarray.open_dataset(options.clay_content)
    model_lat = numpy.arange(-44.0, -9.975, 0.05)
    model_lon = numpy.arange(112.0, 154.025, 0.05)
    clay_content = clay_content.interp(latitude=model_lat, longitude=model_lon)
    clay_content['latitude'].attrs['units'] = 'degrees_north'
    clay_content['latitude'].attrs['axis'] = 'Y'
    clay_content['longitude'].attrs['units'] = 'degrees_east'
    clay_content['longitude'].attrs['axis'] = 'X'
    utils.save_to_netcdf(clay_content, 'results/clay_content_temp.nc')


def splice_maps(options):
    # Set up new dataset to hold result
    latitude = numpy.arange(-44.0, -9.975, 0.05)
    longitude = numpy.arange(112.0, 154.025, 0.05)
    green_date_spliced = xarray.Dataset(
        data_vars={'green_date': (['latitude', 'longitude'], numpy.full([len(latitude), len(longitude)], numpy.nan))},
        coords={'latitude': latitude, 'longitude': longitude}
    )

    clay_content = xarray.open_dataset('results/clay_content_temp.nc')

    with xarray.open_dataset(options.green_date_files.format(threshold='10')) as green_date_10:
        index = clay_content.clay_content_percentage < 20
        green_date_spliced.green_date.values[index] = green_date_10.green_dates.values[index]

    with xarray.open_dataset(options.green_date_files.format(threshold='20')) as green_date_20:
        index = numpy.logical_and(clay_content.clay_content_percentage >= 20, clay_content.clay_content_percentage < 30)
        green_date_spliced.green_date.values[index] = green_date_20.green_dates.values[index]

    with xarray.open_dataset(options.green_date_files.format(threshold='30')) as green_date_30:
        index = numpy.logical_and(clay_content.clay_content_percentage >= 30, clay_content.clay_content_percentage < 35)
        green_date_spliced.green_date.values[index] = green_date_30.green_dates.values[index]

    with xarray.open_dataset(options.green_date_files.format(threshold='40')) as green_date_40:
        index = numpy.logical_and(clay_content.clay_content_percentage >= 35, clay_content.clay_content_percentage < 45)
        green_date_spliced.green_date.values[index] = green_date_40.green_dates.values[index]

    with xarray.open_dataset(options.green_date_files.format(threshold='50')) as green_date_50:
        index = clay_content.clay_content_percentage >= 45
        green_date_spliced.green_date.values[index] = green_date_50.green_dates.values[index]

    utils.save_to_netcdf(green_date_spliced, 'results/green_date_soil_combined.nc')


if __name__ == '__main__':
    main()
