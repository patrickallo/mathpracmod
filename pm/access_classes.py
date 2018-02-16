"""
Module for mixin classes for accessor methods
common to comment_thread and multi_comment_thread
"""

import argparse
import datetime
import logging
from os import getcwd, path, remove
import sys
import yaml
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

LOCATION = path.realpath(path.join(getcwd(), path.dirname(__file__)))


# helper functions
def check_date_type(*args):
    """Checks if dates are of type datetime.datetime,
    and applies strptime if it fails.
    Returns list of datetime.datetime objects"""
    output = []
    for a_date in args:
        try:
            a_date = a_date if isinstance(
                a_date, datetime.datetime) else datetime.datetime.strptime(
                    a_date, "%Y-%m-%d")
            output.append(a_date)
        except ValueError as err:
            print(err, ": datetime failed")
        except TypeError as err:
            print(err, ": datetime failed with ", a_date)
    return output


def color_list(data, vmin, vmax,
               factor=25, cmap=plt.cm.Set1):
    """Input is either int or list-like"""
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    c_mp = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    if isinstance(data, int):
        colors = [c_mp.to_rgba(i * factor) for i in range(data)]
    else:
        try:
            colors = c_mp.to_rgba(data)
        except (ValueError, TypeError):
            logging.warning("Input has to be int or list-like")
            colors = [c_mp.to_rgba(i * factor) for i in range(len(data))]
    return colors


def fake_legend(sizes, title, fun=None, alpha=1, loc=4):
    """Helper-function to create a fake legend for sizes in scatter-plots"""
    if len(sizes) > 3:
        logging.warning("Ignoring remaining sizes")
    if fun:
        scaled_sizes = [fun(size) for size in sizes]
    else:
        scaled_sizes = sizes
    mark1 = plt.scatter([], [], s=scaled_sizes[0], marker='o',
                        color='#555555', alpha=alpha)
    mark2 = plt.scatter([], [], s=scaled_sizes[1], marker='o',
                        color='#555555', alpha=alpha)
    mark3 = plt.scatter([], [], s=scaled_sizes[2], marker='o',
                        color='#555555', alpha=alpha)
    plt.legend((mark1, mark2, mark3),
               sizes, scatterpoints=1,
               loc=loc, borderpad=1.5, labelspacing=2,
               ncol=3, fontsize=8,
               title=title)


get_first_v = np.vectorize(lambda x: x[0])


get_last_v = np.vectorize(lambda x: x[-1])


get_len_v = np.vectorize(len)


to_days = np.vectorize(lambda x: x.total_seconds() / (60**2 * 24))


def handle_delete(filename):
    """Delete file wrapped in try/except"""
    try:
        remove(filename)
    except IOError:
        pass
    else:
        logging.info("Deleting %s", filename)


def load_settings():
    """Load settings from yaml and return 2 dicts"""
    settings_file = path.join(LOCATION, "settings/settings.yaml")
    try:
        with open(settings_file, "r") as settings_file:
            settings = yaml.safe_load(settings_file.read())
            colors1 = getattr(plt.cm, settings['cmap'])(range(20))
            colors2 = getattr(plt.cm, settings['cmap'] + "b")(range(20))
            colors3 = getattr(plt.cm, settings['cmap'] + "c")(range(20))
            colors = np.vstack((colors1, colors2, colors3))
            mymap = mpl.colors.LinearSegmentedColormap.from_list(
                'my_colormap', colors)
    except IOError:
        logging.warning("Could not load settings.")
        sys.exit(1)
    else:
        return settings, mymap, LOCATION


def load_yaml(*args):
    """"Load yaml-files and return as list of dicts"""
    output = []
    for fileref in args:
        fileref = path.join(LOCATION, fileref)
        try:
            with open(fileref, "r") as yaml_file:
                a_dict = yaml.safe_load(yaml_file.read())
        except IOError:
            logging.warning("Could not load date from %s", fileref)
            a_dict = {}
        finally:
            output.append(a_dict)
    return output


def make_arg_parser(actions, project, description):
    """Create and return argparse-object"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("project", nargs="?", default=project,
                        help="Short name of the project")
    parser.add_argument("--more", type=str,
                        choices=actions,
                        help="Show output instead of returning object")
    parser.add_argument("-l", "--load", action="store_true",
                        help="Load serialized threads when available")
    parser.add_argument("-c", "--cache", action="store_true",
                        help="Serialize threads if possible")
    parser.add_argument("-v", "--verbose", type=str,
                        choices=['debug', 'info', 'warning'],
                        default="warning",
                        help="Show more logging information")
    parser.add_argument("-d", "--delete", action="store_true",
                        help="Delete requests and serialized threads")
    return parser


def handle_kwargs(**kwargs):
    """Helper-function to pop some standard kwargs"""
    project = kwargs.pop('project', None)
    show = kwargs.pop('show', True)
    fontsize = kwargs.pop('fontsize', 6)
    return project, show, fontsize


def scale_weights(graph, in_weight, out_weight):
    """Scales edge-weights to unit-interval,
    and adds the scaled weights as separate edge-attributes."""
    as_matrix = nx.to_numpy_matrix(graph, nodelist=None, weight=in_weight)
    scaler = MinMaxScaler()
    as_matrix = scaler.fit_transform(as_matrix.flatten()).reshape(
        as_matrix.shape)
    try:
        assert np.all(as_matrix == as_matrix.T)
    except AssertionError:
        raise RuntimeError("Weight-date improperly scaled")
    weight_data = DataFrame(
        as_matrix, index=graph.nodes(), columns=graph.nodes())
    for source, dest, data in graph.edges(data=True):
        data[out_weight] = weight_data.loc[source, dest]
    return graph


def show_or_save(show):
    """Shows or saves plot"""
    if show:
        plt.show()
    else:
        filename = input("Give filename: ")
        filename += ".png"
        plt.savefig(filename)


def to_pickle(an_object, filename):
    """Pickle and save object wrapped in try/except."""
    try:
        joblib.dump(an_object, filename)
    except RecursionError as err:
        logging.warning("Could not pickle %s: %s", filename, err)
        handle_delete(filename)
    else:
        logging.info("%s saved", filename)


# Mixin Classes
class ThreadAccessMixin(object):
    """description"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_name = {}

    def comment_report(self, com_id):
        """Takes node-id, and returns dict with report about node."""
        the_node = self.graph.node[com_id]  # dict
        the_author = the_node["com_author"]  # string
        descendants = nx.descendants(self.graph, com_id)
        pure_descendants = [i for i in descendants if
                            self.graph.node[i]['com_author'] != the_author]
        direct_descendants = self.graph.out_degree(com_id)
        return {
            "author": the_author,
            "level of comment": the_node["com_depth"],
            "replies (direct)": direct_descendants,
            "replies (all)": len(descendants),
            "replies (own excl.)": len(pure_descendants)
        }

    def print_nodes(self, *select):
        """takes nodes-id(s), and prints out node-data as yaml. No output."""
        if select:
            select = list(self.node_name.keys()) if select[0].lower() == "all"\
                else select
            for comment in select:  # do something if comment does not exist!
                print("com_id:", comment)
                try:
                    print(yaml.dump(self.graph.node[comment],
                                    default_flow_style=False))
                except KeyError as err:
                    print(err, "not found")
                print("------------------------------------------------------")
        else:
            print("No nodes were selected")
