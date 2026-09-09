from dataclasses import dataclass,field
from time import time


@dataclass
class Proposal:
    title:str
    reason:str
    expected_gain:str
    risk:str="unknown"
    created_at:float=field(default_factory=time)


@dataclass
class Candidate:
    proposal:Proposal
    status:str="proposed"
    benchmark_before:float=0.0
    benchmark_after:float=0.0
    approved:bool=False
