import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from beachbums.persons import Person

logger = logging.getLogger(__name__)

rng = np.random.default_rng()


def create_groups_based_on_background(
    persons: dict[str, Person],
    background: str,
    n: int,
    match: bool = True,
    n_shifts: Optional[int] = None,
    save: Optional[Path | str] = None,
):
    """Create groups (size n) of people based on background skill.

    Parameters
    ----------
    persons : List[Person]
        List of Person objects
    background : str
        Which background to group on
    group_size : int
        Number of people in each group
    match : bool, optional
        Match people based on inverse background skill. People with high skill on the
        background are likely to be paired with low skill. False is similar skill.
    n_shifts : int, optional
        Number of random shifts to make to the groups, by default num people / 4.
        This is to make sure that the groups are not always the same.
        Must be smaller than number of people.
    save : Optional[Union[Path, str]], optional
        Save seating plan to file, by default None

    Returns
    -------
    str
        Seating plan as a string
    """

    logger.info("creating groups based on background")
    # get list of people with background skill
    people_with_background = [
        p for p in persons.values() if background in p.backgrounds
    ]
    if len(people_with_background) == 0:
        raise ValueError(
            f"No people with background {background}. Is there a column with this name?"
        )
    logger.info(f"{len(people_with_background)=}")

    # create list of n groups with group_size people in each group
    groups = []
    # sort people by background skill in ascending order
    people_with_background.sort(key=lambda x: x.backgrounds[background])

    if n_shifts is None or n_shifts >= 1:
        # do some random shiftings (by 1) to make sure that the groups are not
        # always the same
        if n_shifts is None:
            n_shifts = len(people_with_background) // 4

        indices_to_shift = rng.choice(
            list(range(len(people_with_background))),
            size=n_shifts,
            replace=False,
        )
        if max(indices_to_shift) == len(people_with_background) - 1:
            idx = np.where(indices_to_shift == max(indices_to_shift))[0][0]
            indices_to_shift[idx] = indices_to_shift[idx] - 1

        for idx in indices_to_shift:
            people_with_background[idx], people_with_background[idx + 1] = (
                people_with_background[idx + 1],
                people_with_background[idx],
            )

    # split group size to evenly take from start and end of list.
    # if group_size is odd, take one more from start (lower skill)
    # so that an expert is more likely to teach 2 novices than 1 novice
    # learning from 2 experts
    take_end = n // 2
    take_start = n - take_end

    # create groups
    if match:
        # get from start and end of list. Group should still be group_size
        while len(people_with_background) >= n:
            group = []
            # if there are not enough people left to fill group, fill with None
            group.extend(people_with_background[:take_start])
            group.extend(people_with_background[-take_end:])
            groups.append(group)
            people_with_background = people_with_background[take_start:-take_end]
    else:
        while len(people_with_background) >= n:
            group = people_with_background[:n]
            groups.append(group)
            people_with_background = people_with_background[n:]

    # add any leftovers to the first groups
    # if match:
    #   the leftovers have middle skill and the first groups are expert-novice
    # else:
    #   the leftovers have high skill and the first groups are novice-novice
    for idx, person in enumerate(people_with_background):
        groups[idx].append(person)

    # create seating plan
    # create a dataframe from the people dict
    grouping_plan = pd.DataFrame.from_records(groups)
    # seating_plan.index.name = "Table"
    # seating_plan.reset_index(inplace=True)
    # seating_plan.rename(columns={"people": "Name"}, inplace=True)
    if save:
        if not isinstance(save, str):
            file_name = (
                f"group-plan-{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
            )
            # check if file exists
            if Path(file_name).exists():
                # file name that includes time
                file_name = f"group-plan-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"
        else:
            file_name = save
        # export to seating-plan-<date>.csv
        grouping_plan.to_csv(
            file_name,
            sep=";",
            index=False,
        )

    return grouping_plan
