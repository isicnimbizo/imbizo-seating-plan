from __future__ import annotations

import datetime
import logging
from functools import partial
from itertools import product
from pathlib import Path
from typing import Callable, Union

import numpy as np
import pandas as pd

from .persons import Person
from .tables import Table

logger = logging.getLogger(__name__)


def process_previous_pairings(
    seating_plans: list[pd.DataFrame], persons: dict[str, Person]
) -> None:
    """
    Add counts of number of times people have sat together

    Args:
        seating_plans: list of seating plans (loaded from seating-plan files)
        persons: dictionary of persons (created from names.csv)
    """
    logger.info("Processing previous pairings")
    for seating_plan in seating_plans:
        groupby_key = "Table" if "Table" in seating_plan else "Group"
        for table_name, table_frame in seating_plan.groupby(groupby_key):
            # list of everyone seated at this table in Person object format
            table_persons_list: list[Person] = []
            for name in table_frame["Name"]:
                if name not in persons:
                    logger.warning(
                        f"{name} has an unknown GROUP, please add them to names.csv"
                    )
                    persons[name] = Person({"NAME": name}, name_col="NAME")
                table_persons_list.append(persons[name])
            # add everyone seated at this table to persons' pairs
            #   note that because pairs are reciprocated (reverse pair is added),
            #   we shorten the list on each iteration
            while len(table_persons_list):
                person = table_persons_list.pop()
                person.add_persons(table_persons_list)


def create_seating_plan(
    people: pd.DataFrame,
    persons: dict[str, Person],
    tables: dict[str, Table],
    name_col="NAME",
    save=True,
    seed=None,
):
    """
    randomize seating based on TAs being at each table and evenly seat people at
    each table

    :param people: dataframe containing GROUP (TA, STUDENT, or FACULTY) and people's names
    :param tables: dict of dict with table name: {size, people}
    """
    logger.info("Creating seating plan")
    rng = np.random.default_rng(seed) if seed else None
    # filter people based on EXCLUDE column
    excluded_rows = people[(people["EXCLUDE"] == "X") | (people["EXCLUDE"] == "Y")]
    excluded_names: list[str] = list(excluded_rows[name_col].values)
    people = people[~people["NAME"].isin(excluded_names)]

    tas_s = people[people["GROUP"] == "TA"][name_col]
    random_tas: list[str] = tas_s.sample(frac=1, random_state=rng).tolist()
    ta_persons = {name: persons[name] for name in random_tas}

    students_df = people[people["GROUP"] == "STUDENT"]
    students_df_preferred_mask = students_df["PREFERRED"].isna()
    random_students: list[str] = (
        students_df[students_df_preferred_mask][name_col]
        .sample(frac=1, random_state=rng)
        .tolist()
    )
    students_with_preferences: list[str] = students_df[students_df_preferred_mask][
        name_col
    ].tolist()
    student_persons = {
        name: persons[name] for name in students_with_preferences + random_students
    }

    faculty_s = people[people["GROUP"] == "FACULTY"][name_col]
    random_faculty: list[str] = faculty_s.sample(frac=1, random_state=rng).tolist()
    faculty_persons = {name: persons[name] for name in random_faculty}

    # first, sort tables by size (using value of sub-dict key "capacity") with smallest
    # tables first
    sorted_tables = dict(sorted(tables.items(), key=lambda x: x[1].capacity))
    # then, combine people with TAs first, then FACULTY, then STUDENTS
    unseated_people = ta_persons | faculty_persons | student_persons

    # ensure everyone is an option
    for ta, other_person in product(
        ta_persons.values(), (student_persons | faculty_persons).values()
    ):
        ta.add_option(other_person)

    # keep track of where a person is sitting
    people_table_placement: dict[str, Union[int, Table]] = {
        person_name: -1 for person_name in unseated_people.keys()
    }

    # create a partial function to add a person to a table, taking in the people dict and
    # table dict
    add_person_to_table_func = add_person_to_table_factory(
        people_table_placement, tables
    )

    # count size of tables
    total_seats = sum([table.capacity for table in sorted_tables.values()])
    if total_seats < len(unseated_people):
        raise ValueError(
            f"Not enough seats for all people! {total_seats} < {len(unseated_people)}"
        )
    extra_seats = total_seats - len(unseated_people)
    extra_tas = {}
    if len(ta_persons) > (num_sorted_tables := len(sorted_tables)):
        extra_tas = {
            name: ta_persons.pop(name)
            for name in list(ta_persons.keys())[num_sorted_tables:]
        }

    # seat TAs
    for (table_name, table), (ta_name, ta) in zip(
        sorted_tables.items(), ta_persons.items()
    ):
        added = add_person_to_table_func(ta, table_name)
        if not added:
            raise ValueError(f"Could not add {ta} to {table_name}")

    # for each person, get the normalised count of people of people at each table
    # and choose the lowest
    for group in [faculty_persons, student_persons, extra_tas]:
        for person_name, person in group.items():
            table_vals: dict[str, float] = {}
            for table_name, table in sorted_tables.items():
                # ignore tables that are full
                if table.is_full:
                    continue
                options, counts = person.get_pair_count_for_people(table.people)
                for option in options:
                    if option.preferred == person_name:
                        table_vals[table_name] = 0
                if person.is_faculty:
                    for i, option in enumerate(options):
                        if option.is_faculty:
                            if counts[i] == 0:
                                counts[i] = 100
                            else:
                                counts[i] *= 100
                table_vals[table_name] = (table.seated / table.capacity) + (
                    sum(counts) / len(options)
                )
            # choose the table with the lowest value and that
            min_table_name = min(table_vals, key=table_vals.get)  # type: ignore
            added = add_person_to_table_func(person, min_table_name)
            if logger.getEffectiveLevel() <= logging.DEBUG:
                pretty_tables = "\n".join(
                    [
                        f"{table_vals.get(table_name,-1):.2f}-{table}"
                        for table_name, table in tables.items()
                    ]
                )
                logger.debug(f"added {person} to {min_table_name}.\n{pretty_tables}")
            if not added:
                if tables[min_table_name].is_full:
                    raise ValueError(f"Table '{min_table_name}' is full.")
                elif people_table_placement[person_name] != -1:
                    raise ValueError(f"Person '{person_name}' already seated.")
                raise ValueError(f"Could not add {person} to any table")

    unseated_people = {
        person: persons[person]
        for person in unseated_people
        if people_table_placement[person] == -1
    }
    if unseated_people:
        raise ValueError(f"Could not seat everyone! {unseated_people}")

    # now, loop through the remaining people (including faculty) and
    # assign them to tables
    table_names = list(sorted_tables.keys())
    table_name = table_names[0]
    for person_name, person in unseated_people.items():
        # assign starting point for checking in the while loop
        table_start = table_name
        while not add_person_to_table_func(person, table_name):
            # loop through tables until we find one that works
            table_name = table_names[
                (table_names.index(table_name) + 1) % len(table_names)
            ]
            if table_name == table_start:
                raise ValueError(f"Could not seat {person}, tried all tables")
        table_name = table_names[(table_names.index(table_name) + 1) % len(table_names)]

    # checks
    # 1. check if faculty are distributed to each table
    faculty_table_count = {table_name: 0 for table_name in table_names}
    for table_name, table in sorted_tables.items():
        faculty_table_count[table_name] = len(
            [person for person in table.people if person.group == "FACULTY"]
        )
    # count the tables with more than 1 faculty
    multi_faculty_table_count = [
        table_name for table_name, count in faculty_table_count.items() if count > 1
    ]
    # count the tables with 0 faculty
    zero_faculty_table_count = [
        table_name for table_name, count in faculty_table_count.items() if count == 0
    ]
    if len(multi_faculty_table_count) > 0 and len(zero_faculty_table_count) > 0:
        # log the tables with more than 1 faculty
        err_msg = (
            f"Some tables have more than one faculty and some tables have no faculty."
            f"\nMulti: {multi_faculty_table_count} and Zero:{zero_faculty_table_count}"
        )
        logger.error(err_msg)
        for table_name, table in sorted_tables.items():
            if table_name in multi_faculty_table_count:
                logger.error(f"MULTI -> {table_name}: {table.people}")
            elif table_name in zero_faculty_table_count:
                logger.error(f"ZERO  -> {table_name}: {table.people}")
            else:
                logger.warning(f" OK -> {table_name}: {table.people}")
        raise ValueError(err_msg)

    # convert to pure string; use sub-dict for easier creation of dataframe
    tables_dict = {k: {"people": [p.name for p in t.people]} for k, t in tables.items()}

    # create a dataframe from the table dict
    seating_plan = pd.DataFrame.from_dict(tables_dict, orient="index").explode("people")
    seating_plan.index.name = "Table"
    seating_plan.reset_index(inplace=True)
    seating_plan.rename(columns={"people": "Name"}, inplace=True)
    if save:
        if not isinstance(save, str):
            file_name = (
                f"seating-plan-{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
            )
            # check if file exists
            if Path(file_name).exists():
                # file name that includes time
                file_name = f"seating-plan-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"
        else:
            file_name = save
        # export to seating-plan-<date>.csv
        seating_plan.to_csv(
            file_name,
            sep=";",
            index=False,
        )

    return seating_plan


def add_person_to_table_factory(
    people_table_placement: dict[str, int | Table], table_options: dict[str, Table]
) -> Callable[[Person, str], bool]:
    return partial(_add_person_to_table_generic, people_table_placement, table_options)


def _add_person_to_table_generic(
    people_table_placement: dict[str, int | Table],
    table_options: dict[str, Table],
    person: Person,
    table_name: str,
) -> bool:
    """
    Add a person to a table.
    """
    if (
        table_options[table_name].has_space
        and people_table_placement[person.name] == -1
    ):
        table_options[table_name].add_person(person)
        people_table_placement[person.name] = table_options[table_name]
        return True
    return False


def ta_centric_algorithm(
    excluded_persons,
    faculty_persons,
    ta_persons,
    sorted_tables,
    people_table_placement,
    extra_seats,
    extra_tas,
    rng,
):
    """
    Algorithm 2: seat TAs and select students based on frequency of sitting with the TA
    """
    add_person_to_table_func = add_person_to_table_factory(
        people_table_placement, sorted_tables
    )

    # first, seat TAs
    all_excluded = set(excluded_persons + faculty_persons + ta_persons)
    # assign tas to table and choose people they haven't sat with yet
    for (table_name, table), ta in zip(sorted_tables.items(), ta_persons):
        # add TA
        added = add_person_to_table_func(ta, table_name)
        if not added:
            raise ValueError(f"Could not add {ta} to {table_name}")
        # get previous pairs
        options, counts = ta.get_pair_count_for_everyone_except(all_excluded)
        # add 1 to denominator so that inverse is 1 not 0
        adjusted_counts = np.array(counts) + 1
        # power to make it even less likely to sit with the same person
        p = 1 / adjusted_counts**4
        # assign a probability of 0 to people already at tables
        for i, option in enumerate(options):
            if people_table_placement[option.name] != -1:
                p[i] = 0
        # make sure p sums to 1
        p = p / p.sum()

        # people to seat.
        # if there are extra seats, don't fill up the tables initially. This would mean
        # the last table could be empty.
        size = (
            table.capacity
            - table.seated
            - int(
                np.ceil(
                    (extra_seats + len(faculty_persons) + len(extra_tas))
                    / len(sorted_tables),
                )
            )
        )
        # make sure
        if size > np.sum(p > 0):
            size = np.sum(p > 0)
        # choose some people
        choices = rng.choice(
            options,
            size=size,
            replace=False,
            p=p,
        )
        # add them to the table
        for choice in choices:
            added = add_person_to_table_func(choice, table_name)
            if not added:
                raise ValueError(f"Could not add {choice} to {table_name}")

    # remove assigned people from list of people to seat
    unseated_people = [
        person for person in unseated_people if people_table_placement[person] == -1
    ]
    unseated_persons = [persons[person] for person in unseated_people]
