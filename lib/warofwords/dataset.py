import json

from dataclasses import dataclass
from typing import FrozenSet


@dataclass(eq=True, frozen=True)
class Edit:
    id: int
    i1: int
    i2: int
    j1: int
    j2: int
    tag: str


@dataclass(eq=True, frozen=True)
class Author:
    id: int
    nationality: str
    group: str
    rapporteur: bool


@dataclass(eq=True, frozen=True)
class Merge:
    """Class describing info about merged edits."""
    edit: Edit
    document_ref: str
    amendment_type: str


@dataclass(eq=True, frozen=True)
class Datum:
    accepted: bool
    edit: Edit
    merged_with: FrozenSet[Merge]
    authors: FrozenSet[Author]
    article_type: str
    amendment_type: str
    document_ref: str
    dossier_ref: str
    text_org: list = None
    text_amd: list = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'edit':
                object.__setattr__(self, 'edit', Edit(**v))
            elif k == 'merged_with' and v is not None:
                merges = [Merge(edit=Edit(**d['edit']),
                                document_ref=d['document_ref'],
                                amendment_type=d['amendment_type']) for d in v]
                object.__setattr__(
                    self, 'merged_with', frozenset(merges))
            elif k == 'authors':
                object.__setattr__(
                    self, 'authors', frozenset([Author(**a) for a in v]))
            else:
                self.__dict__[k] = v


class Dataset:

    @staticmethod
    def load(path):
        with open(path) as f:
            lines = f.readlines()
        if type(json.loads(lines[0])) is list:
            return [[Datum(**c) for c in json.loads(s)] for s in lines]
        else:
            return [Datum(**json.loads(s)) for s in lines]

    def load_json(path):
        with open(path) as f:
            return [json.loads(l) for l in f.readlines()]
