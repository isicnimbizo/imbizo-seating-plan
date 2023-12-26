from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from beachbums.persons import Person

logger = logging.getLogger(__name__)


def create_groups_based_on_background(
    persons: dict[str, Person],
    background: str,
    group_size: int,
    match: bool = True,
    save: Optional[Path | str] = None,
    seed: Optional[int] = None,
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
    save : Optional[Union[Path, str]], optional
        Save seating plan to file, by default None

    Returns
    -------
    str
        Seating plan as a string
    """

    logger.info("creating groups based on background")
    rng = np.random.default_rng(seed)

    # get list of people with background skill
    people_with_background = [
        p for p in persons.values() if background in p.backgrounds
    ]
    if len(people_with_background) == 0:
        raise ValueError(
            f"No people with background {background}. Is there a column with this name?"
        )
    logger.info(f"{len(people_with_background)=}")

    # sort people by background skill in ascending order
    people_with_background.sort(key=lambda x: x.backgrounds[background])

    # for each person, select someone with oppositite background and not paired before
    groups: list[list[Person]] = []
    while len(people_with_background) >= group_size:
        person = people_with_background.pop()
        logger.debug(f"Considering {person.name}...")
        group = [person]

        # get background skill of person
        person_background_skill = person.backgrounds[background]
        # get pairs
        pairs, counts = person.get_pair_count_for_people(people_with_background)
        counts = np.array(counts)
        # get background skill of pairs
        pair_background_skills = [p.backgrounds[background] for p in pairs]
        if match:
            # if match, then we want to match people with high skill with low skill
            skill_diff = np.power(
                np.array(pair_background_skills) - person_background_skill, 2
            )
        else:
            # if not match, then we want to match people with similar skill
            # (we start selecting from high skill)
            skill_diff = np.array(pair_background_skills) + person_background_skill

        # adjust by pair counts
        skill_diff = np.clip(skill_diff - max(skill_diff) / 2 * counts**2, 0, None)

        # sort by skill diff
        idx_sort = np.argsort(skill_diff)
        # reorder
        skill_diff = skill_diff[idx_sort]
        pairs = [pairs[i] for i in idx_sort]
        counts = counts[idx_sort]

        if np.all(skill_diff == 0):
            # if all skill diff is 0, then set all to 1 for uniform probability
            skill_diff = np.ones_like(skill_diff)

        # select with weighted probability
        other_people_in_group = rng.choice(
            pairs,
            size=group_size - 1,
            replace=False,
            p=skill_diff / skill_diff.sum(),
        )
        # remove from list
        for idx in other_people_in_group:
            people_with_background.remove(idx)

        group.extend(other_people_in_group)
        groups.append(group)

    if len(people_with_background) > len(groups):
        raise ValueError(
            f"Number of people with background ({len(people_with_background)}) "
            f"is larger than number of groups ({len(groups)}). "
            f"Decrease group size or increase number of people."
        )
    # add any leftovers to the first groups
    # if match:
    #   the leftovers have middle skill and the first groups are expert-novice
    # else:
    #   the leftovers have high skill and the first groups are novice-novice
    idx = 0
    while len(people_with_background) > 0:
        groups[idx].append(people_with_background.pop())
        idx += 1

    # convert to pure string (p.name); use sub-dict for easier creation of dataframe
    grouping_plan = pd.DataFrame.from_dict(
        {
            f"Group {g}": {"people": [p.name for p in people]}
            for g, people in enumerate(groups)
        },
        orient="index",
    ).explode("people")
    grouping_plan.index.name = "Group"
    grouping_plan.reset_index(inplace=True)
    grouping_plan.rename(columns={"people": "Name"}, inplace=True)
    if save:
        if not isinstance(save, str):
            file_name = f"group-plan-{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
            # check if file exists
            if Path(file_name).exists():
                # file name that includes time
                file_name = f"group-plan-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"
        else:
            file_name = save

        # sort by number
        parse_skip_n_chars = len("Group ")
        grouping_plan["number"] = grouping_plan["Group"].apply(
            lambda x: int(x[parse_skip_n_chars:])
        )
        grouping_plan.sort_values(by="number", inplace=True)
        grouping_plan.drop(columns="number", inplace=True)

        # export to group-plan-<date>.csv
        grouping_plan.to_csv(
            file_name,
            sep=";",
            index=False,
        )

    return grouping_plan
