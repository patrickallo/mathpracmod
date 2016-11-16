"""Function to access parts of the PM_FRAME"""
# imports
import logging

import numpy as np
import pandas as pd

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


def get_project_at(pm_frame, project, thread_type, stage):
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


def thread_or_project(pm_frame, project, thread_type, stage, thread):
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
