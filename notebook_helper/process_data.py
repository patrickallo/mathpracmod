"""
Module with convenience-functions to create a DataFrame with
Data about all Polymath-projects.
Intended use: within notebook.
"""
# importing modules
from collections import Counter, OrderedDict
import logging
import pandas as pd
from pandas import DataFrame, Series
# importing own modules
import comment_thread as ct
import author_network as an
import multi_comment_thread as mct


# function to create DataFrame with all data for single project
def create_project_frame(project, titles, split):
    """Creates empty DataFrame with MultiIndex for index and columns
    Arguments: project-name, list of thread-titles, split(True/False)
    Return: DataFrame, list of indices for the threads"""
    indices = list(range(1, len(titles) + 1))
    cols_0 = (7 * ['basic'] +
              10 * ['all threads'] +
              10 * ['research threads'] +
              10 * ['discussion threads'])
    cols_1 = (['title', 'url', 'blog', 'research', 'thread',
               'post length', 'avg length of comments'] +
              3 * ['mthread (single)', 'mthread (accumulated)',
                   'number of comments', 'number of comments (accumulated)',
                   'network',
                   'authors', 'number of authors', 'authors (accumulated)',
                   'comment_counter', 'comment_counter (accumulated)'])
    assert len(cols_0) == len(cols_1)
    if split:
        cols = pd.MultiIndex.from_arrays([cols_0, cols_1])
    else:
        cols = pd.MultiIndex.from_arrays([cols_0[:15], cols_1[:15]])
    rows = pd.MultiIndex.from_tuples([(project, i) for i in indices],
                                     names=['Project', 'Ord'])
    return DataFrame(index=rows, columns=cols), indices


# function to add data to cols based on thread-types
def split_thread_types(pm_frame):
    """Fills the additional columns for research and discussion threads.
    Argument: DataFrame with intended columns.
    Return: DataFrame with additional data"""
    r_indices = pm_frame[pm_frame['basic', 'research']].index
    d_indices = pm_frame[~pm_frame['basic', 'research']].index
    r_threads = pm_frame[pm_frame['basic', 'research']]['basic', 'thread']
    d_threads = pm_frame[~pm_frame['basic', 'research']]['basic', 'thread']
    pm_frame['research threads', 'mthread (single)'] = pm_frame[
        pm_frame['basic', 'research']]['all threads', 'mthread (single)']
    pm_frame['research threads', 'mthread (accumulated)'] = Series(
        [mct.MultiCommentThread(
            *r_threads.loc[:, pd.IndexSlice[:i]]) for
            i in r_indices.droplevel()],
        index=r_indices)
    pm_frame['research threads', 'network'] = pm_frame[
        'research threads', 'mthread (accumulated)'].dropna().apply(
            an.AuthorNetwork)
    pm_frame['discussion threads', 'mthread (single)'] = pm_frame[
        ~pm_frame['basic', 'research']]['all threads', 'mthread (single)']
    pm_frame['discussion threads', 'mthread (accumulated)'] = Series(
        [mct.MultiCommentThread(
            *d_threads.loc[:, pd.IndexSlice[:i]]) for
            i in d_indices.droplevel()],
        index=d_indices)
    pm_frame['discussion threads', 'network'] = pm_frame[
        'discussion threads', 'mthread (accumulated)'].dropna().apply(
            an.AuthorNetwork)
    return pm_frame


# function to add comment and author-data
def extend_project_frame(pm_frame):
    """Fills the columns with info about comments and authors.
    Argument: DataFrame with intended columns.
    Return: DataFrame with additional data"""
    for thread_type in ['all threads', 'research threads',
                        'discussion threads']:
        try:
            pm_frame[thread_type, 'number of comments'] = pm_frame[
                thread_type, 'mthread (single)'].dropna().apply(
                    lambda x: len(x.node_name.keys()))
            pm_frame[
                thread_type, 'number of comments (accumulated)'] = pm_frame[
                    thread_type, 'mthread (accumulated)'].dropna().apply(
                        lambda x: len(x.node_name.keys()))
            pm_frame[thread_type, 'authors'] = pm_frame[
                thread_type, 'mthread (single)'].dropna().apply(
                    lambda mthread: set(mthread.author_color.keys()))
            pm_frame[thread_type, 'number of authors'] = pm_frame[
                thread_type, 'authors'].dropna().apply(len)
            pm_frame[thread_type, 'authors (accumulated)'] = pm_frame[
                thread_type, 'mthread (accumulated)'].dropna().apply(
                    lambda mthread: set(mthread.author_color.keys()))
            pm_frame[thread_type, 'comment_counter'] = pm_frame[
                thread_type, 'mthread (single)'].dropna().apply(
                    lambda mthread: Counter(mthread.node_name.values()))
            pm_frame[thread_type, 'comment_counter (accumulated)'] = pm_frame[
                thread_type, 'mthread (accumulated)'].dropna().apply(
                    lambda mthread: Counter(mthread.node_name.values()))
        except KeyError:  # when split=False not all types are in pm_frame
            pass
    return pm_frame


# function for forward-filling nan's where needed
def fill_project_frame(pm_frame):
    """forward-fills NaN-values for author/comment-related cols
    Argument: DataFrame with intended columns.
    Return: DataFrame with additional data"""
    for thread_type in ['research threads', 'discussion threads']:
        for data_type in ['number of comments (accumulated)',
                          'authors (accumulated)',
                          'comment_counter (accumulated)']:
            pm_frame[thread_type, data_type] = pm_frame[
                thread_type, data_type].fillna(method='ffill')
    return pm_frame


# function to process a single polymath-project (relies on above functions)
def process_polymath(project, split=False):
    """Created DataFrame for a given project
    Argument: Project-title.
    Return: DataFrame with additional data"""
    titles, threads = list(zip(*ct.main(
        project.replace("Polymath", "pm"),
        use_cached=False, cache_it=False, merge=False).items()))
    pm_frame, indices = create_project_frame(project, titles, split)
    pm_frame['basic', 'title'] = titles
    pm_frame['basic', 'thread'] = threads
    pm_frame['basic', 'url'] = [thread.data.url for thread in threads]
    pm_frame['basic', 'research'] = [
        thread.data.is_research for thread in threads]
    pm_frame['basic', 'blog'] = [
        thread.data.thread_url.netloc.split('.')[0].title() for
        thread in threads]

    pm_frame['all threads', 'mthread (single)'] = [
        mct.MultiCommentThread(thread) for thread in threads]
    pm_frame['all threads', 'mthread (accumulated)'] = [
        mct.MultiCommentThread(*threads[0:i]) for i in indices]
    pm_frame['all threads', 'network'] = pm_frame[
        'all threads', 'mthread (accumulated)'].apply(an.AuthorNetwork)
    pm_frame['basic', 'post length'] = pm_frame['basic', 'thread'].apply(
        lambda x: len(x.post_content.split()))
    pm_frame['basic', 'avg length of comments'] = pm_frame[
        'all threads', 'mthread (single)'].dropna().apply(
            lambda x: x.count_activity()['wordcounts'].mean())
    if split:
        pm_frame = split_thread_types(pm_frame)
    pm_frame = pm_frame.pipe(
        extend_project_frame).pipe(
            fill_project_frame)
    return pm_frame


# function to process all projects within chosen range, and return as dict
def process_pms(*args):
    """Takes any number of ints as argument, and returns dict of
    project-name: DataFrame by calling process_polymath"""
    out = OrderedDict()
    for arg in args:
        name = "Polymath {}".format(arg)
        out[name] = process_polymath(name, split=True)
        logging.info("%s processed", name)
    return out


def concatenate_project_dfs(dct):
    """Takes dict of project-dataframes and returns concatenation."""
    polymaths = list(dct.keys())
    col_order = dct[polymaths[0]].columns.tolist()
    return pd.concat(dct.values())[col_order]
