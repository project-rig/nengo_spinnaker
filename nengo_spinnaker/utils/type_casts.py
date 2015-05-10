"""Standard typecasts for Nengo on SpiNNaker."""
import rig.type_casts as tp


value_to_fix = tp.float_to_fix(True, 32, 15)  # Float -> S16.15
fix_to_value = tp.fix_to_float(True, 32, 15)  # S16.15 -> Float

np_to_fix = tp.NumpyFloatToFixConverter(True, 32, 15)  # Float -> S16.15
fix_to_np = tp.NumpyFixToFloatConverter(15)  # xX.15 -> Float
