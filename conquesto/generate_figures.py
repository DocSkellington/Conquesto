""" Copyright © - 2020 - UMONS
    CONQUESTO of University of Mons - Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet - is free software : you can redistribute it and/or modify it under the terms of the BSD-3 Clause license. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the BSD-3 Clause License for more details. 
    You should have received a copy of the BSD-3 Clause License along with this program. 
    Each use of this software must be attributed to University of Mons (Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet).
"""
from enum import Enum
from typing import Tuple, List, Dict
import pickle
import math
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt

import generate_statistics as statistics
import query.query as query
from generate_files import load_queries

class Color(Enum):
    GC = 'blue'
    FO = 'red'
    OTHER = 'dimgray'

def load_dataset(filepath : str) -> Tuple[statistics.Description, statistics.Dataset]:
    with open(filepath, "r") as f:
        # The first line is the description
        line = f.readline().strip()
        splitted = line.split(" ")
        if splitted[0] == "yes":
            database_type = statistics.DatabaseType.YES
        elif splitted[0] == "no":
            database_type = statistics.DatabaseType.NO
        elif splitted[0] == "random":
            database_type = statistics.DatabaseType.RANDOM
        else:
            raise ValueError("{} is not a known database type".format(splitted[0]))
        time_limit = int(splitted[1])
        number_of_queries = int(splitted[2])
        number_of_tables = int(splitted[3])
        positions_per_table = tuple([int(x) for x in splitted[4:]])
        description = statistics.Description(database_type, time_limit, number_of_queries, number_of_tables, positions_per_table)

        dataset : statistics.Dataset = []

        # The first line is already read!
        # The remaining lines are the dataset
        for line in f:
            splitted = line.strip().split(" ")
            size_parameter = int(splitted[0])
            actual_size = int(splitted[1])
            query_id = int(splitted[2])

            gc_timeout = splitted[3] == "True"
            gc_satisfiable = splitted[4] == "True"
            gc_cpu_time = float(splitted[5])
            gc_choices = int(splitted[6])
            gc_rules = int(splitted[7])
            gc_atoms = int(splitted[8])
            gc_bodies = int(splitted[9])
            gc_variables = int(splitted[10])
            gc_constraints = int(splitted[11])
            gc = statistics.Results(gc_timeout, gc_cpu_time, gc_satisfiable, gc_choices, gc_rules, gc_atoms, gc_bodies, gc_variables, gc_constraints)

            fo_timeout = splitted[12] == "True"
            fo_satisfiable = splitted[13] == "True"
            fo_cpu_time = float(splitted[14])
            fo_choices = int(splitted[15])
            fo_rules = int(splitted[16])
            fo_atoms = int(splitted[17])
            fo_bodies = int(splitted[18])
            fo_variables = int(splitted[19])
            fo_constraints = int(splitted[20])
            fo = statistics.Results(fo_timeout, fo_cpu_time, fo_satisfiable, fo_choices, fo_rules, fo_atoms, fo_bodies, fo_variables, fo_constraints)

            dataset.append(statistics.Line(size_parameter, actual_size, query_id, gc, fo))


    return (description, dataset)

def from_varying_to_fixed_database(dataset : statistics.Dataset, parameter_value : int) -> statistics.Dataset:
    return list(filter(lambda l: l.sizeParameter == parameter_value, dataset))

def compute_means_variances_varying_database_size(dataset : statistics.Dataset, actual_size_as_axis : bool, min_x_value : int) -> Tuple[List[int], np.array, np.array, np.array, np.array]:
    mean_vals_gc_dict : Dict[int, float] = {}
    mean_vals_fo_dict : Dict[int, float] = {}

    var_vals_gc_dict : Dict[int, float] = {}
    var_vals_fo_dict : Dict[int, float] = {}

    for line in dataset:
        if actual_size_as_axis:
            n = line.actualSize
            def comp(l):
                return l.actualSize == n
        else:
            n = line.sizeParameter
            def comp(l):
                return l.sizeParameter == n
        if n < min_x_value:
            continue

        if n not in mean_vals_fo_dict:
            times = list(map(lambda l: (l.gc.cpu_time * 1000, l.fo.cpu_time * 1000), filter(comp, dataset)))
            gc_times = [x[0] for x in times]
            fo_times = [x[1] for x in times]
            mean_vals_gc_dict[n] = np.mean(gc_times)
            mean_vals_fo_dict[n] = np.mean(fo_times)
            var_vals_gc_dict[n] = np.var(gc_times)
            var_vals_fo_dict[n] = np.var(fo_times)

    def to_list(d: Dict[int, float]) -> List[float]:
        return list(map(lambda t: t[1], sorted(d.items(), key=lambda k: k[0])))
    def get_x(d: Dict[int, float]) -> List[int]:
        return list(map(lambda t: t[0], sorted(d.items(), key=lambda k: k[0])))
        
    x_vals = get_x(mean_vals_gc_dict)
    mean_vals_gc = to_list(mean_vals_gc_dict)
    mean_vals_fo = to_list(mean_vals_fo_dict)
    var_vals_gc = to_list(var_vals_gc_dict)
    var_vals_fo = to_list(var_vals_fo_dict)
    np_mean_vals_gc = np.array(mean_vals_gc)
    np_mean_vals_fo = np.array(mean_vals_fo)
    np_var_vals_gc = np.sqrt(np.array(var_vals_gc))
    np_var_vals_fo = np.sqrt(np.array(var_vals_fo))

    return x_vals, np_mean_vals_gc, np_mean_vals_fo, np_var_vals_gc, np_var_vals_fo

def compute_medians_first_quartiles_varying_database_size(dataset : statistics.Dataset, actual_size_as_axis : bool, min_x_value : int) -> Tuple[List[int], np.array, np.array, np.array, np.array]:
    median_vals_gc_dict : Dict[int, float] = {}
    median_vals_fo_dict : Dict[int, float] = {}
    first_vals_gc_dict : Dict[int, float] = {}
    first_vals_fo_dict : Dict[int, float] = {}

    for line in dataset:
        if actual_size_as_axis:
            n = line.actualSize
            def comp(l):
                return l.actualSize == n
        else:
            n = line.sizeParameter
            def comp(l):
                return l.sizeParameter == n
        if n < min_x_value:
            continue

        if n not in median_vals_gc_dict:
            times = list(map(lambda l: (l.gc.cpu_time * 1000, l.fo.cpu_time * 1000), filter(comp, dataset)))
            gc_times = [x[0] for x in times]
            fo_times = [x[1] for x in times]
            median_vals_gc_dict[n] = np.median(gc_times)
            median_vals_fo_dict[n] = np.median(fo_times)
            first_vals_gc_dict[n] = np.quantile(gc_times, 0.25)
            first_vals_fo_dict[n] = np.quantile(fo_times, 0.25)

    def to_list(d: Dict[int, float]) -> List[float]:
        return list(map(lambda t: t[1], sorted(d.items(), key=lambda k: k[0])))
    def get_x(d: Dict[int, float]) -> List[int]:
        return list(map(lambda t: t[0], sorted(d.items(), key=lambda k: k[0])))
        
    x_vals = get_x(median_vals_gc_dict)
    median_vals_gc = to_list(median_vals_gc_dict)
    median_vals_fo = to_list(median_vals_fo_dict)
    first_vals_gc = to_list(first_vals_gc_dict)
    first_vals_fo = to_list(first_vals_fo_dict)
    np_median_vals_gc = np.array(median_vals_gc)
    np_median_vals_fo = np.array(median_vals_fo)
    np_first_vals_gc = np.sqrt(np.array(first_vals_gc))
    np_first_vals_fo = np.sqrt(np.array(first_vals_fo))

    return x_vals, np_median_vals_gc, np_median_vals_fo, np_first_vals_gc, np_first_vals_fo

def generate_time_graphic_varying_database_size(description: statistics.Description, data: statistics.Dataset, save_file: str):
    '''
    Generates a graphic with the size of the database as the X axis and the time taken as the Y axis.
    The dataset must contain at most one line by size!

    :param description: The description of the dataset.
    :param data: The dataset.
    :param save_file: Where to save the figure.
    '''
    x_vals : List[int] = []
    y_vals_gc : List[float] = []
    y_vals_fo : List[float] = []

    for n, _, _, gc, fo in data:
        x_vals.append(n)
        y_vals_gc.append(gc.cpu_time)
        y_vals_fo.append(fo.cpu_time)

    fig, ax = plt.subplots(figsize = (10, 10))

    g_gc, = ax.plot(x_vals, y_vals_gc, "-o", color = Color.GC.value)
    g_gc.set_label("Generate and check program")
    g_fo, = ax.plot(x_vals, y_vals_fo, "-o", color = Color.FO.value)
    g_fo.set_label("FO program")

    instance = "yes" if description.databaseType else "no"
    ax.legend()
    ax.set_title("Time taken on %s databases of various sizes" % instance)
    ax.set_xlabel("The size of the parameter n")
    ax.set_ylabel("The time taken (in seconds)")
    ax.grid()

    fig.savefig(save_file)

def generate_mean_satisfiable_graphic_varying_database_size(description : statistics.Description, dataset : statistics.Dataset, min_x_value : int, actual_size_as_axis : bool, save_file : str):
    '''
    Generates a graphic with the size of the database as the X axis and the mean satisfiability as the Y axis.

    :param description: The description of the dataset.
    :param dataset: The dataset.
    :param min_x_value: The minimal value in the X axis.
    :param actual_size_as_axis: Whether to use the actual size of the database in the X axis or the parameter controlling the size.
    :param save_file: Where to save the figure.
    '''
    mean_satisfiable_gc_dict : Dict[int, float] = {}
    mean_satisfiable_fo_dict : Dict[int, float] = {}

    for line in dataset:
        if actual_size_as_axis:
            n = line.actualSize
            def comp(l):
                return l.actualSize == n
        else:
            n = line.sizeParameter
            def comp(l):
                return l.sizeParameter == n

        if n < min_x_value:
            continue

        if n not in mean_satisfiable_gc_dict:
            sat = list(map(lambda l: (int(l.gc.satisfiable), int(l.fo.satisfiable)), filter(comp, dataset)))
            gc_sat = [x[0] for x in sat]
            fo_sat = [x[1] for x in sat]
            mean_satisfiable_gc_dict[n] = np.mean(gc_sat)
            mean_satisfiable_fo_dict[n] = np.mean(fo_sat)

    def to_list(d: Dict[int, float]) -> List[float]:
        return list(map(lambda t: t[1], sorted(d.items(), key=lambda k: k[0])))
    def get_x(d: Dict[int, float]) -> List[int]:
        return list(map(lambda t: t[0], sorted(d.items(), key=lambda k: k[0])))
        
    x_vals = get_x(mean_satisfiable_gc_dict)
    mean_vals_gc = to_list(mean_satisfiable_gc_dict)
    mean_vals_fo = to_list(mean_satisfiable_fo_dict)
    np_mean_vals_gc = np.array(mean_vals_gc)
    np_mean_vals_fo = np.array(mean_vals_fo)

    fig, ax = plt.subplots(figsize = (10, 10))

    g_gc, = ax.plot(x_vals, np_mean_vals_gc, "o", color = Color.GC.value)
    g_gc.set_label("Generate and check program")

    g_fo, = ax.plot(x_vals, np_mean_vals_fo, "o", color = Color.FO.value)
    g_fo.set_label("FO program")

    ax.legend()
    if actual_size_as_axis:
        ax.set_xlabel("The actual size of the database")
    else:
        ax.set_xlabel("The value of the parameter controlling the size of the database")
    ax.set_ylabel("The average of satisfiability (1 means satisfiable; 0 means unsatisfiable)")
    ax.grid()

    fig.savefig(save_file)

def generate_mean_time_graphic_varying_database_size(description : statistics.Description, dataset : statistics.Dataset, min_x_value : int, actual_size_as_axis : bool, save_file : str, with_variance : bool = True):
    '''
    Generates a graphic with the size of the database as the X axis and the mean time taken as the Y axis.

    :param description: The description of the dataset.
    :param dataset: The dataset.
    :param min_x_value: The minimal value in the X axis.
    :param actual_size_as_axis: Whether to use the actual size of the database in the X axis or the parameter controlling the size.
    :param save_file: Where to save the figure.
    :param with_variance: Whether to show the variance in the figure.
    '''
    x_vals, np_mean_vals_gc, np_mean_vals_fo, np_var_vals_gc, np_var_vals_fo = compute_means_variances_varying_database_size(dataset, actual_size_as_axis, min_x_value)

    fig, ax = plt.subplots(figsize = (10, 10))

    g_gc, = ax.plot(x_vals, np_mean_vals_gc, "-o", color = Color.GC.value)
    g_gc.set_label("Generate and test")
    if with_variance:
        ax.plot(x_vals, np_mean_vals_gc + np_var_vals_gc, "-", color = Color.GC.value)
        ax.plot(x_vals, np_mean_vals_gc - np_var_vals_gc, "-", color = Color.GC.value)
        ax.fill_between(x_vals, np_mean_vals_gc + np_var_vals_gc, np_mean_vals_gc - np_var_vals_gc, facecolor=Color.GC.value, alpha=0.3, interpolate = True)

    g_fo, = ax.plot(x_vals, np_mean_vals_fo, "-o", color = Color.FO.value)
    g_fo.set_label("First-order rewriting")
    if with_variance:
        ax.plot(x_vals, np_mean_vals_fo + np_var_vals_fo, "-", color = Color.FO.value)
        ax.plot(x_vals, np_mean_vals_fo - np_var_vals_fo, "-", color = Color.FO.value)
        ax.fill_between(x_vals, np_mean_vals_fo + np_var_vals_fo, np_mean_vals_fo - np_var_vals_fo, facecolor=Color.FO.value, alpha=0.3, interpolate = True)

    ax.legend()
    if actual_size_as_axis:
        ax.set_xlabel("The actual size of the database")
    else:
        ax.set_xlabel("Parametric database size")
    ax.set_ylabel("Average CPU time (in milliseconds)")
    ax.grid()

    fig.savefig(save_file, bbox_inches="tight")

def generate_mean_median_time_graphic_fixed_database_size(description : statistics.Description, dataset : statistics.Dataset, min_parameter_value : int, max_parameter_value : int, save_file : str):
    gc_at_least_one_timeout = fo_at_least_one_timeout = False
    gc_poks : Dict[int, float] = {}
    fo_poks : Dict[int, float] = {}
    for value in range(min_parameter_value, max_parameter_value + 1):
        data = from_varying_to_fixed_database(dataset, value)
        gc_timeouts, fo_timeouts = number_of_timeouts(data)
        if gc_timeouts != 0:
            gc_at_least_one_timeout = True
        if fo_timeouts != 0:
            fo_at_least_one_timeout = True
        gc_pok = (description.numberOfQueries - gc_timeouts) / description.numberOfQueries * 100
        fo_pok = (description.numberOfQueries - fo_timeouts) / description.numberOfQueries * 100
        gc_poks[value] = gc_pok
        fo_poks[value] = fo_pok

    x_vals, np_mean_vals_gc, np_mean_vals_fo, _, _ = compute_means_variances_varying_database_size(dataset, False, 0)
    _, np_median_vals_gc, np_median_vals_fo, np_first_vals_gc, np_first_vals_fo = compute_medians_first_quartiles_varying_database_size(dataset, False, 0)

    fig, ax = plt.subplots(figsize = (12, 12))

    g_gc_mean, = ax.plot(x_vals, np_mean_vals_gc, "-o", color = Color.GC.value)
    if gc_at_least_one_timeout:
        g_gc_mean.set_label("Generate and test (lower bound for the mean)")
    else:
        g_gc_mean.set_label("Generate and test (mean)")
    g_gc_median, = ax.plot(x_vals, np_median_vals_gc, "-+", color = Color.GC.value)
    g_gc_median.set_label("Generate and test (median)")

    g_fo_mean, = ax.plot(x_vals, np_mean_vals_fo, "-o", color = Color.FO.value)
    if fo_at_least_one_timeout:
        g_fo_mean.set_label("First-order rewriting (lower bound for the mean)")
    else:
        g_fo_mean.set_label("First-order rewriting (mean)")

    if not (np_mean_vals_fo == np_median_vals_fo).all():
        # If the mean and the median are disjoint
        g_fo_median, = ax.plot(x_vals, np_median_vals_fo, "-+", color = Color.FO.value)
        g_fo_median.set_label("First-order rewriting (median)")

    ax.legend(loc="upper left")
    ax.set_xlabel("Parametric database size")
    ax.set_ylabel("CPU time (in milliseconds)")
    ax.grid()
    fig.savefig(save_file, bbox_inches="tight")

def generate_mean_database_size_graphic_varying_database_size(description : statistics.Description, dataset : statistics.Dataset, min_x_value : int, save_file : str, with_variance : bool = True):
    '''
    Generates a graphic with the value of the parameter controlling the size of the database as the X axis and the actual database size as the Y axis.

    :param description: The description of the dataset.
    :param dataset: The dataset.
    :param min_x_value: The minimal value in the X axis.
    :param save_file: Where to save the figure.
    :param with_variance: Whether to show the variance.
    '''
    mean_vals_dict : Dict[int, int] = {}

    var_vals_dict : Dict[int, int] = {}

    for line in dataset:
        n = line.sizeParameter
        if n < min_x_value:
            continue

        if n not in mean_vals_dict:
            size = list(map(lambda l: l.actualSize, filter(lambda l: n == l.sizeParameter, dataset)))
            mean_vals_dict[n] = np.mean(size)
            var_vals_dict[n] = np.var(size)

    def to_list(d: Dict[int, int]) -> List[int]:
        return list(map(lambda t: t[1], sorted(d.items(), key=lambda k: k[0])))
    def get_x(d: Dict[int, int]) -> List[int]:
        return list(map(lambda t: t[0], sorted(d.items(), key=lambda k: k[0])))
        
    x_vals = get_x(mean_vals_dict)
    mean_vals = to_list(mean_vals_dict)
    var_vals = to_list(var_vals_dict)
    np_mean_vals = np.array(mean_vals)
    np_var_vals = np.sqrt(np.array(var_vals))

    fig, ax = plt.subplots(figsize = (10, 10))

    g_size, = ax.plot(x_vals, np_mean_vals, "-o", color=Color.OTHER.value)
    if with_variance:
        ax.plot(x_vals, np_mean_vals + np_var_vals, "-", color = Color.OTHER.value)
        ax.plot(x_vals, np_mean_vals - np_var_vals, "-", color = Color.OTHER.value)
        ax.fill_between(x_vals, np_mean_vals + np_var_vals, np_mean_vals - np_var_vals, facecolor=Color.OTHER.value, alpha=0.3, interpolate = True)

    ax.set_xlabel("Parametric database size")
    ax.set_ylabel("Average number of facts in the database")
    ax.grid()

    fig.savefig(save_file, bbox_inches="tight")


def generate_time_graphic_fixed_database_size(description : statistics.Description, dataset : statistics.Dataset, parameter_value : int, save_file : str, logarithmic_scale : bool = False):
    '''
    Generates a graphic with a boxplot for the GC programs and a boxplot for the FO programs.
    The Y axis is the time taken.

    :param description: The description of the dataset.
    :param data: The dataset.
    :param parameter_value: The value of the database size parameter for which to plot.
    :param save_file: Where to save the figure.
    :param logarithmic_scale: Whether to use a logarithmic scale (log10) for the Y axis
    '''
    fixed_dataset = from_varying_to_fixed_database(dataset, parameter_value)
    y_vals_gc = list(map(lambda l: l.gc.cpu_time * 1000, fixed_dataset))
    y_vals_fo = list(map(lambda l: l.fo.cpu_time * 1000, fixed_dataset))
    if logarithmic_scale:
        y_vals_gc = np.log10(y_vals_gc)
        y_vals_fo = np.log10(y_vals_fo)

    fig, ax = plt.subplots(figsize = (10, 10))

    median_style = {"color": Color.OTHER.value, "lw": 5, "solid_capstyle": "round"}
    bp = ax.boxplot([y_vals_gc, y_vals_fo], positions = [1, 2], autorange=False, widths=0.4, labels = ["Generate and test", "First-order rewriting"], medianprops=median_style, patch_artist=True)

    colors = [Color.GC.value, Color.FO.value]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)

    if logarithmic_scale:
        ax.set_ylabel("CPU time (in logarithmic milliseconds)")
    else:
        ax.set_ylabel("CPU time (in milliseconds)")
    ax.grid()

    fig.savefig(save_file, bbox_inches="tight")

def _update_query_list_normalized(worst_queries : List[Tuple[query.Query, float, int, int]], best_queries : List[Tuple[query.Query, float, int, int]], query : query.Query, time : float, keep_queries : int, parameter_size : int, database_size : int):
    '''
    Updates the best and worst queries lists.

    :param worst_list: The list of worst queries.
    :param best_list: The list of best queries.
    :param query: The current query.
    :param time: The time taken to solve the current query.
    :param keep_queries: The maximum size of the two lists.
    :param database_size: The size of database; used to normalize.
    '''

    #update worst time list
    if len(worst_queries) > 0 and len(worst_queries) >= keep_queries:
        worst_time_indice = np.argmin([x[1] for x in worst_queries]) #actually the best of the worst times
        old_time = worst_queries[worst_time_indice][1] 
        if old_time < time:
            worst_queries.append((query, time, parameter_size, database_size)) #if worst than the one we found, append the query to the list
        if len(worst_queries) > keep_queries:
            del worst_queries[worst_time_indice] #if the list is full delete the best element
    else:
        worst_queries.append((query, time, parameter_size, database_size))

    #update best time list
    if len(best_queries) > 0 and len(best_queries) >= keep_queries:
        best_time_indice = np.argmax([x[1] for x in best_queries]) #actually the worst of the best times
        old_time = worst_queries[best_time_indice][1]
        if old_time < time:
            best_queries.append((query, time, parameter_size, database_size)) #if better than the one we found, append the query to the list
        if len(best_queries) > keep_queries:
            del best_queries[best_time_indice] #if the list is full  delete the worst element
    else:
        best_queries.append((query, time, parameter_size, database_size))

def find_best_and_worst_queries(queries : List[query.Query],
                                dataset : statistics.Dataset,
                                n_queries : int,
                                with_timeouts : bool = False
                               ) -> Tuple[List[Tuple[query.Query, float, int, int]], List[Tuple[query.Query, float, int, int]], List[Tuple[query.Query, float, int, int]], List[Tuple[query.Query, float, int, int]]]:
    '''
    Finds the n_queries best and worst queries, according to the CPU time in the dataset.

    :param queries: The list of queries.
    :param dataset: The dataset.
    :param n_queries: The number of queries to keep.
    :param with_timeouts: If False, timeouts are discarded.

    :return: The lists of best and worst queries for GC, and the lists for FO.
    '''
    best_queries_gc : List[Tuple[query.Query, float, int, int]] = []
    worst_queries_gc : List[Tuple[query.Query, float, int, int]] = []
    best_queries_fo : List[Tuple[query.Query, float, int, int]] = []
    worst_queries_fo : List[Tuple[query.Query, float, int, int]] = []

    for parameter_size, actual_size, query_id, gc, fo in dataset:
        gc_time = gc.cpu_time
        fo_time = fo.cpu_time
        query = queries[query_id]
        if with_timeouts or not gc.timeout:
            _update_query_list_normalized(worst_queries_gc, best_queries_gc, query, gc_time, n_queries, parameter_size, actual_size)
            _update_query_list_normalized(worst_queries_fo, best_queries_fo, query, fo_time, n_queries, parameter_size, actual_size)

    return (best_queries_gc, worst_queries_gc, best_queries_fo, worst_queries_fo)

def sort_by_time(queries : List[query.Query], dataset : statistics.Dataset) -> Tuple[List[Tuple[query.Query, float, int, int]], List[Tuple[query.Query, float, int, int]]]:
    # We group by parametric size and sort by CPU time
    gc = sorted(map(lambda l: (queries[l.queryID], l.gc.cpu_time, l.sizeParameter, l.actualSize), dataset), key=itemgetter(2, 1))
    fo = sorted(map(lambda l: (queries[l.queryID], l.fo.cpu_time, l.sizeParameter, l.actualSize), dataset), key=itemgetter(2, 1))
    return gc, fo

def number_of_timeouts(dataset : statistics.Dataset) -> Tuple[int, int]:
    '''
    Gives the number of timeouts for GC and FO in the given dataset.

    :param dataset: The dataset

    :return: The number of timeouts for GC and for FO
    '''
    n_timeouts_gc = 0
    n_timeouts_fo = 0
    for _, _, _, gc, fo in dataset:
        if gc.timeout:
            n_timeouts_gc += 1
        if fo.timeout:
            n_timeouts_fo += 1
    return (n_timeouts_gc, n_timeouts_fo)

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 22})
    extension = "png"
    
    timelimit = 100

    program = statistics.Program.DLV.value.lower()

    n_queries = 194

    fixed_database_size = False
    min_n = 40
    max_n = 70

    table_number = 2

    queries = load_queries("generated/queries_{}_{}_{}.pickle".format(table_number, n_queries, fixed_database_size))

    datasets : List[Tuple[statistics.Description, statistics.Dataset]] = []
    for db_type in [statistics.DatabaseType.YES]:
        if fixed_database_size:
            description, dataset = load_dataset("generated/dataset_fixed_{}_{}_{}_{}_{}_{}_{}.txt".format(program, db_type.value, table_number, fixed_database_size, max_n, len(queries), timelimit))
        else:
            description, dataset = load_dataset("generated/dataset_varying_{}_{}_{}_{}_{}_{}.txt".format(db_type.value.lower(), program, min_n, max_n, n_queries, timelimit))

        datasets.append((description, dataset))

        print("For {}".format(db_type.value)) 
        gc, fo = sort_by_time(queries, dataset)
        print("\tGC:")
        for q, time, parameter, size in gc:
            print("\t", q, time, parameter, size)
        print("\tFO:")
        for q, time, parameter, size in fo:
            print("\t", q, time, parameter, size)

        # We want the queries that timed out at min_n
        at_min_n = from_varying_to_fixed_database(dataset, min_n)
        gc_at_min_n = filter(lambda l: l[2], map(lambda l: (queries[l.queryID], l.gc.cpu_time, l.gc.timeout, l.sizeParameter), at_min_n))
        fo_at_min_n = filter(lambda l: l[2], map(lambda l: (queries[l.queryID], l.fo.cpu_time, l.fo.timeout, l.sizeParameter), at_min_n))
        print("\tAt {}".format(min_n))
        print("\t\tGC:")
        for q, time, timedout, parameter in gc_at_min_n:
            print("\t\t", q, time, timedout, parameter)
        print("\t\tFO:")
        for q, time, timedout, parameter in fo_at_min_n:
            print("\t\t", q, time, timedout, parameter)

        # We do the same at max_n
        at_max_n = from_varying_to_fixed_database(dataset, max_n)
        gc_at_max_n = filter(lambda l: l[2], map(lambda l: (queries[l.queryID], l.gc.cpu_time, l.gc.timeout, l.sizeParameter), at_max_n))
        fo_at_max_n = filter(lambda l: l[2], map(lambda l: (queries[l.queryID], l.fo.cpu_time, l.fo.timeout, l.sizeParameter), at_max_n))
        print("\tAt {}".format(max_n))
        print("\t\tGC:")
        for q, time, timedout, parameter in gc_at_max_n:
            print("\t\t", q, time, timedout, parameter)
        print("\t\tFO:")
        for q, time, timedout, parameter in fo_at_max_n:
            print("\t\t", q, time, timedout, parameter)


    # First, we generate boxplots for a specific parameter value
    # fixed_n = max_n
    # for description, dataset in datasets:
    #     generate_time_graphic_fixed_database_size(description, dataset, fixed_n, "generated/figures/fixed_time_{}_{}_{}_{}_{}_{}.{}".format(description.databaseType.value, table_number, fixed_database_size, fixed_n, len(queries), timelimit, extension))

    # Then, the full plots for all values
    # actual_size_as_axis = False
    # with_variance = not actual_size_as_axis
    # if actual_size_as_axis:
    #     min_x_value = 4000
    # else:
    #     min_x_value = 0

    for description, dataset in datasets:
        # generate_mean_time_graphic_varying_database_size(description, dataset, min_x_value, actual_size_as_axis, "generated/figures/mean_time_{}_{}_{}_{}_{}_{}_{}_{}.{}".format(program, description.databaseType.value, table_number, fixed_database_size, min_n, max_n, len(queries), timelimit, extension), with_variance)
        # generate_mean_database_size_graphic_varying_database_size(description, dataset, min_x_value, "generated/figures/mean_size_{}_{}_{}_{}_{}_{}_{}_{}.{}".format(program, description.databaseType.value, table_number, fixed_database_size, min_n, max_n, len(queries), timelimit, extension), with_variance)
        generate_mean_median_time_graphic_fixed_database_size(description, dataset, min_n, max_n, "generated/figures/mean_median_time_{}_{}_{}_{}_{}_{}.{}".format(program, description.databaseType.value, min_n, max_n, len(queries), timelimit, extension))

        # if description.databaseType == statistics.DatabaseType.RANDOM:
        #     generate_mean_satisfiable_graphic_varying_database_size(description, dataset, min_n, False, "generated/figures/mean_satisfiable_{}_{}_{}_{}_{}_{}_{}.{}".format(program, description.databaseType.value, table_number, fixed_database_size, max_n, len(queries), timelimit, extension))
