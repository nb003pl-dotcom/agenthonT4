"""Optional lossless input wire for joint relevance-and-fact calls."""
from pathlib import Path

from .predict_input_wire import encode_input_wire

ROOT=Path(__file__).resolve().parents[1]


def apply(system,payload,mode):
    if mode not in ('original-json','compact-xml'):
        raise ValueError('unknown joint input wire: '+str(mode))
    wire,audit=encode_input_wire(payload,mode)
    if audit is None:return system,wire,None
    guide=(ROOT/'prompts/joint-evidence-input-wire-v1.md').read_text()
    return system.rstrip()+'\n\n'+guide.strip()+'\n',wire,audit
