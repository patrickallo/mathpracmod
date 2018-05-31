"""Function to access parts of the PM_FRAME"""
# imports
from functools import partial
import logging

import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import author_network as an


# utility-functions to get specific parts of PM_FRAME
def get_last(pm_frame, thread_type):
    """thread_type is either 'all threads', 'research threads',
    or 'discussion threads'
    Returns tuple of:
    (1) DataFrame with only last row for each project
    (2) List of positions for last rows DataFrame"""
    source = pm_frame[
        ['basic', thread_type]].dropna() if thread_type else pm_frame
    projects = source.index.droplevel(1).unique()
    ords = [source.loc[project].index[-1] for project in projects]
    lasts_index = pd.MultiIndex.from_tuples(list(zip(projects, ords)),
                                            names=['Project', 'Ord'])
    data = source.loc[lasts_index]
    return data, lasts_index


def get_project_at(pm_frame, project, thread_type, stage, **kwargs):
    """Returns sub-row of PM_FRAME based on project, thread_type and stage"""
    if stage == -1:
        logging.info("Stage is last non-null stage of %s", project)
        out = pm_frame.loc[project][thread_type].dropna().iloc[stage]
    else:
        out = pm_frame.loc[project][thread_type].iloc[stage]
        nulls = pd.isnull(out)
        if np.any(nulls):
            logging.warning("Some empty values: %s", out.index[nulls])
    return out


def thread_or_project(pm_frame, project, thread_type, stage, thread, **kwargs):
    """Helper function to create data from project or single thread."""
    if thread is None:
        data = get_project_at(pm_frame, project, thread_type, stage)['network']
    else:
        print("Overriding thread_type and stage")
        mthread = get_project_at(
            pm_frame, project, "all threads", thread)['mthread (single)']
        thread_title = get_project_at(pm_frame, project,
                                      "basic", thread)['title']
        data = an.AuthorNetwork(mthread)
        project = "{}\n{}".format(project, thread_title)
    return project, data

# utility-functions used in Analysis to get and clean specific data


def get_graph_and_safely_remove_anon(
        an_netw, graph_type="c_dgraph", remove_anon=True):
    """Takes an AuthorNetwork, and returns a chosen author-graph type from it
    after (optionally) removing Anonymous."""
    the_graph = getattr(an_netw, graph_type).copy()
    if remove_anon:
        try:
            the_graph.remove_node("Anonymous")
        except nx.NetworkXError:
            pass
        else:
            print("x", end="")
        finally:
            assert "Anonymous" not in the_graph
    return the_graph


def get_final_networks(pm_frame, project):
    """Takes name of project, and extracts the networks from a pre-existing
    dataframe with project-data (in global variable)."""
    author_network = get_project_at(
        pm_frame, project, 'all threads', stage=-1).network
    networks = dict()
    for graph_type in ["i_graph", "c_graph", "c_dgraph"]:
        networks[graph_type] = get_graph_and_safely_remove_anon(
            author_network, graph_type=graph_type)
        if networks[graph_type].is_directed():
            largest_component = sorted(nx.weakly_connected_components(
                networks[graph_type]), key=len)[-1]
        else:
            largest_component = sorted(nx.connected_components(
                networks[graph_type]), key=len)[-1]
        core = graph_type + "_core"
        networks[core] = networks[graph_type].subgraph(largest_component)
    return author_network, networks


def assemble_project_evolution_data(
        pm_frame, project, graph_type, thread_type="all threads"):
    project_data = pm_frame.loc[project]
    evolution = project_data[thread_type, 'network'].apply(partial(
        get_graph_and_safely_remove_anon, graph_type=graph_type))
    return evolution


def evolution_over_time(
        pm_frame, project, final_author_network=None,
        thread_type="all threads", top_n=5):
    if final_author_network is None:
        final_author_network = get_project_at(
            pm_frame, project, thread_type, stage=-1).network
    top = final_author_network.author_frame['total comments'].nlargest(
        top_n).index
    evolution_cd = assemble_project_evolution_data(
        pm_frame, project, "c_dgraph", thread_type)
    evolution_i = assemble_project_evolution_data(
        pm_frame, project, "i_graph", thread_type)
    evol_data = DataFrame(index=evolution_cd.index)
    for (g_type, evolution) in zip(
            ["cluster", "interaction"], [evolution_cd, evolution_i]):
        sizes = []
        cc_sizes = []
        max_degree = []
        u_p_l = []
        diameters = []
        degree_top_c = {key: [] for key in top}
        for graph in evolution:
            if graph.is_directed:
                graph = graph.copy().to_undirected()
            sizes.append(graph.number_of_nodes())
            max_degree.append(Series(dict(graph.degree())).max())
            for key, value in degree_top_c.items():
                value.append(graph.degree(key))
            largest_cc = max(nx.connected_components(graph), key=len)
            sub_graph = graph.subgraph(largest_cc)
            cc_sizes.append(sub_graph.number_of_nodes())
            diameters.append(nx.diameter(sub_graph))
            u_p_l.append(nx.average_shortest_path_length(sub_graph))
        if g_type == "cluster":
            evol_data["N"] = sizes
        evol_data["N* ({})".format(g_type)] = cc_sizes
        evol_data["max degree ({})".format(g_type)] = max_degree
        evol_data["avg shortest path-length ({})".format(g_type)] = u_p_l
        for key, value in degree_top_c.items():
            try:
                evol_data["{} ({})".format(key, g_type)] = value
            except KeyError:
                print(evol_data.index)
                print(value)
                print(len(value))
    mask = evol_data.applymap(lambda x: not isinstance(x, (int, float)))
    evol_data[mask] = np.nan
    evol_data["ln(N)"] = np.log(evol_data["N"])
    evol_data["ln(N*) (cluster)"] = np.log(evol_data["N* (cluster)"])
    evol_data["ln(N*) (interaction)"] = np.log(evol_data["N* (interaction)"])
    evol_data["lnln(N)"] = np.log(np.log(evol_data["N"]))
    evol_data["lnln(N*) (cluster)"] = np.log(np.log(evol_data["N* (cluster)"]))
    evol_data["lnln(N*) (interaction)"] = np.log(
        np.log(evol_data["N* (interaction)"]))
    return evol_data, top


def assemble_indiv_networks(
        pm_frame, project, thread_type="all threads", thresh=2):
    project_data = pm_frame.loc[project]
    indiv_networks = project_data[
        thread_type, "mthread (single)"].dropna().apply(an.AuthorNetwork)
    indiv_networks = indiv_networks.apply(get_graph_and_safely_remove_anon)
    degree_data = indiv_networks.apply(
        lambda x: Series(dict(x.out_degree(weight="weight"))))
    degree_data_core = degree_data.dropna(axis=1, thresh=thresh)
    core = degree_data_core.columns
    return {"networks": indiv_networks,
            "core": core,
            "thread_type": thread_type}
