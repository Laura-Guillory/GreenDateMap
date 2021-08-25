import logging
import xarray
import cartopy
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot
from cartopy.io import shapereader
import warnings
import argparse
from datetime import datetime


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

    gen_map(options)

    end_time = datetime.now()
    LOGGER.info('End time: ' + str(end_time))
    elapsed_time = end_time - start_time
    LOGGER.info('Elapsed time: ' + str(elapsed_time))


def get_options():

    """
    Gets command line arguments and returns them.
    Options are accessed via options.verbose, etc.

    Optional arguments: verbose (v), title

    Run this with the -h (help) argument for more detailed information. (python gen_map.py -h)

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
        '--green_date',
        help='The path to the Green Date results to be mapped.',
        required=True
    )
    parser.add_argument(
        '--output',
        help='The path to save the resulting map at.',
        default='results/green_date.png'
    )
    parser.add_argument(
        '--title',
        help='The title of the map produced. Defaults to no title.',
        default=''
    )
    args = parser.parse_args()
    return args


def gen_map(options):
    """
    Converts the results from netCDF to a png image.

    :param options:
    :return:
    """
    data = xarray.open_dataset(options.green_date)
    projection = cartopy.crs.PlateCarree()
    left, right, bottom, top = 112, 154, -28, -10
    TITLE_FONT = FontProperties(fname='fonts/Roboto-Light.ttf', size=12)
    colours = ['#374a9f', '#3967a3', '#4575b3', '#659bc8', '#8abeda', '#acdae9', '#cfebf3', '#ebf7e4', '#fffebe',
               '#fee99d', '#feca7c', '#fca85e', '#f67a49', '#e54f35', '#d02a27', '#b10b26', '#999999']

    figure = pyplot.figure(figsize=(8, 8))  # Set size of the plot
    # Create axis for the plot using the desired projection and extent
    ax = pyplot.axes(projection=projection, extent=(left, right, bottom, top+2))
    pyplot.gca().outline_patch.set_visible(False)  # Remove border around plot

    # Get the shape reader
    shape = shapereader.Reader('shapes/gadm36_AUS_1.shp')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        # Plot data as a contour. Levels are used to decide the threshold for each contour level (currently 3 levels
        # per month, see the colourbar in the result)
        levels = [30, 40, 50, 61, 71, 81, 91, 101, 111, 122, 132, 142, 153, 163, 173, 181]
        im = ax.contourf(data['longitude'], data['latitude'], data['green_date'], extend='both',
                         transform=cartopy.crs.PlateCarree(), levels=levels, colors=colours, zorder=1)

    # Draw borders
    for state in shape.records():
        ax.add_geometries([state.geometry], cartopy.crs.PlateCarree(), edgecolor='black', facecolor='none',
                          linewidth=0.4, zorder=3)

    # Add towns
    shape_fn = shapereader.natural_earth(resolution='10m', category='cultural', name='populated_places')
    towns = []
    featurecla = ['Admin-0 capital', 'Admin-0 capital alt', 'Admin-0 region capital', 'Admin-1 region capital']
    skip_towns = ['Cloncurry', 'Roebourne', 'McMinns Lagoon', 'Barcaldine', 'Charleville', 'Sunshine Coast', 'Dalby',
                  'Port Douglas', 'Atherton', 'Innisfail', 'Ingham', 'Ayr', 'Charters Towers', 'Proserpine', 'Emerald',
                  'Yeppoon', 'Gladstone', 'Biloela', 'Hervey Bay', 'Maryborough', 'Kingaroy', 'Toowoomba', 'Caloundra',
                  'Bowen', 'Caboolture', 'Bongaree', 'Gympie', 'Moranbah']
    for record in shapereader.Reader(shape_fn).records():
        if record.attributes['ADM0NAME'] == 'Australia' and record.geometry.coords[0][1] > -28 \
                and (record.attributes['POP_MAX'] > 1000 or record.attributes['FEATURECLA'] in featurecla) \
                and not record.attributes['NAME'] in skip_towns:
            towns.append({
                'name': record.attributes['NAME'],
                'x': record.geometry.coords[0][0],
                'y': record.geometry.coords[0][1]
            })
    for town in towns:
        ax.plot('x', 'y', data=town, marker='o', markerfacecolor='white', markeredgewidth=1, markeredgecolor='black',
                markersize=3, zorder=4)
        ax.text(town['x']+.3, town['y'], town['name'], va='center', ha='left', fontsize='x-small')

    # Add a colourbar
    colourbar_axis = figure.add_axes([0.21, 0.25, .6, .02])
    colourbar = figure.colorbar(im, cax=colourbar_axis, extendfrac=.05, orientation='horizontal')
    date_ticklabels = ['1 Oct', '1 Nov', '1 Dec', '1 Jan', '1 Feb', '1 Mar']
    colourbar.set_ticks([30, 61, 91, 122, 153, 181])
    colourbar.set_ticklabels(date_ticklabels)
    for tick in colourbar.ax.get_xticklabels():
        tick.set_font_properties(FontProperties(fname='fonts/Roboto-Light.ttf', size=8))

    # Add the title
    pyplot.text(.3, 1, options.title, transform=ax.transAxes, fontproperties=TITLE_FONT)

    # Save map
    pyplot.savefig(options.output, dpi=400, bbox_inches='tight', pil_kwargs={'quality': 80})
    pyplot.close()


if __name__ == '__main__':
    main()