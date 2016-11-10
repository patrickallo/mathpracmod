"""Function to access parts of the PM_FRAME"""
# imports
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame

PM_FRAME = DataFrame()


# utility-functions to get specific parts of PM_FRAME
def get_last(thread_type):
    """thread_type is either 'all threads', 'research threads',
    or 'discussion threads'
    Returns tuple of:
    (1) DataFrame with only last row for each project
    (2) List of positions for last rows DataFrame"""
    source = PM_FRAME[
        ['basic', thread_type]].dropna() if thread_type else PM_FRAME
    projects = source.index.droplevel(1).unique()
    ords = [source.loc[project].index[-1] for project in projects]
    lasts_index = pd.MultiIndex.from_tuples(list(zip(projects, ords)),
                                            names=['Project', 'Ord'])
    data = source.loc[lasts_index]
    return data, lasts_index


def get_project_at(project, thread_type, stage):
    """Returns sub-row of PM_FRAME based on project, thread_type and stage"""
    if stage == -1:
        logging.info("Stage is last non-null stage of %s", project)
        out = PM_FRAME.loc[project][thread_type].dropna().iloc[stage]
    else:
        out = PM_FRAME.loc[project][thread_type].iloc[stage]
        nulls = pd.isnull(out)
        if np.any(nulls):
            logging.warning("Some empty values: %s", out.index[nulls])
    return out
