import pytest
from beachbums.persons import Person


@pytest.fixture
def sipho_student_record():
    return {
        "NAME": "Sipho",
        "GROUP": "STUDENT",
        "Maths": 10,
        "Physics": 9,
        "Neuroscience": 5,
        "EXCLUDE": float("nan"),
    }


@pytest.fixture
def sipho_person(sipho_student_record):
    return Person(sipho_student_record)


def test_person_init():
    record = {
        "NAME": "Sipho",
        "GROUP": "STUDENT",
        "Math": 10,
        "Physics": 9,
        "Neuro": 5,
        "EXCLUDE": float("nan"),
    }
    person = Person(record)

    assert person.name == "Sipho"
    assert person.group == "STUDENT"
    assert person.backgrounds == {"Math": 10, "Physics": 9, "Neuro": 5}
    assert not person.exclude


@pytest.mark.parametrize(
    ("exclude_value", "exclude_assertion"),
    [
        ("False", False),
        (float("nan"), False),
        ("N", False),
        ("", False),
        ("True", True),
        ("X", True),
        ("Anything", True),
    ],
)
def test_person_exclude(exclude_value, exclude_assertion):
    record = {
        "NAME": "Tata",
        "GROUP": "STUDENT",
        "EXCLUDE": exclude_value,
    }
    person = Person(record)

    assert (
        person.exclude == exclude_assertion
    ), f"Expected {exclude_assertion} from {exclude_value} but got {person.exclude}"


def test_person_repr_str(sipho_person):
    assert repr(sipho_person) == "Sipho"
    assert str(sipho_person) == "Sipho"


def test_person_eq(sipho_student_record):
    person1 = Person(sipho_student_record)
    person2 = Person(sipho_student_record)

    assert person1 == person2


def test_person_hash(sipho_student_record):
    person1 = Person(sipho_student_record)
    person2 = Person(sipho_student_record)

    assert hash(person1) == hash(person2)


def test_person_is_faculty():
    record = {"NAME": "Sipho", "GROUP": "FACULTY"}
    person = Person(record)

    assert person.is_faculty


def test_person_is_ta():
    record = {"NAME": "Sipho", "GROUP": "TA"}
    person = Person(record)

    assert person.is_ta


def test_person_is_student():
    record = {"NAME": "Sipho", "GROUP": "STUDENT"}
    person = Person(record)

    assert person.is_student


def test_person_add_persons():
    record1 = {"NAME": "Sipho"}
    record2 = {"NAME": "Alice"}
    record3 = {"NAME": "Bob"}
    person = Person(record1)
    other_person1 = Person(record2)
    other_person2 = Person(record3)

    person.add_persons([other_person1, other_person2])

    assert person.pair_counts[other_person1] == 1
    assert person.pair_counts[other_person2] == 1


def test_person_add_pair():
    record1 = {"NAME": "Sipho"}
    record2 = {"NAME": "Alice"}
    person = Person(record1)
    other_person = Person(record2)

    person.add_pair(other_person)

    assert person.pair_counts[other_person] == 1
    assert person.get_pair_count_for_person(other_person) == 1
    assert person.pair_counts[Person({"NAME": "Random"})] == 0


def test_person_get_pairs():
    record1 = {"NAME": "Sipho"}
    record2 = {"NAME": "Alice"}
    person = Person(record1)
    other_person = Person(record2)

    person.add_pair(other_person)

    assert list(person.get_pairs()) == [other_person]


def test_person_get_total_pair_count():
    record1 = {"NAME": "Sipho"}
    record2 = {"NAME": "Alice"}
    record3 = {"NAME": "Bob"}
    person = Person(record1)
    other_person1 = Person(record2)
    other_person2 = Person(record3)

    person.add_persons([other_person1, other_person2])

    assert person.get_total_pair_count() == 2


def test_person_get_pair_count_for_person():
    record1 = {"NAME": "Sipho"}
    record2 = {"NAME": "Alice"}
    person = Person(record1)
    other_person = Person(record2)

    person.add_pair(other_person)

    assert person.get_pair_count_for_person(other_person) == 1
