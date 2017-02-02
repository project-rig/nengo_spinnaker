import enum


class OutputPort(enum.Enum):
    """Indicate the intended transmitting part of an executable."""
    standard = 0
    """Standard, value-based, output port."""


class InputPort(enum.Enum):
    """Indicate the intended receiving part of an executable."""
    standard = 0
    """Standard, value-based, output port."""


class EnsembleOutputPort(enum.Enum):
    """Ensemble only output ports."""
    neurons = 0
    """Spike-based neuron output."""

    learnt = 1
    """Output port whose decoders are learnt"""


class EnsembleInputPort(enum.Enum):
    """Ensemble only input ports."""
    neurons = 0
    """Spike-based neuron input."""

    global_inhibition = 1
    """Global inhibition input."""

    learnt = 2
    """Input port whose encoders are learnt"""
