"""Peak-tracking demodulation for Fabry-Perot and fibre Bragg grating sensors.

The package is deliberately flat: detection, fitting and tracking are pure
functions over NumPy arrays, so each can be tested apart; ``io`` is the one
module that touches files, reading the instruments' exports and aligning them
onto one clock.
"""

__version__ = "0.1.0"
