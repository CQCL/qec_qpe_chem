"""
Code for auto generating logical circuits.

# TODO: Fill in papers - Iceberg, Steane & Ciaran & colleagues hardware papers
"""

from .rz_encoding import (
    RzEncoding,
    RzDirect,
    RzNonFt,
    RzFtPrep,
    RzRusNonFt,
    RzMeasFt,
    RzPartFt,
    RzKNonFt,
    RzKMeasFt,
)

from .state_prep import (
    get_non_ft_prep,
    get_ft_prep,
)

from .basic_gates import (
    get_H,
    get_X,
    get_Z,
    get_S,
    get_Sdg,
    get_V,
    get_Vdg,
    get_CX,
    get_Measure,
)

from .steane_corrections import (
    classical_steane_decoding,
    steane_z_correction,
    steane_x_correction,
    steane_lookup_table,
    syndrome_from_readout,
    readout_correction,
)

from .iceberg_detections import (
    iceberg_detect_zx,
    iceberg_detect_z,
    iceberg_detect_x,
)


from .encode import (
    encode,
    RzMode,
    RzOptionsRUS,
    RzOptionsBinFracNonFT,
    RzOptionsBinFracMeasFT,
    RzOptionsBinFracPartFT,
    EncodeData,
    EncodeOptions,
    get_encoded_circuit,
)

from .decode import (
    ReadoutMode,
    InterpretOptions,
    get_decoded_result,
    interpret,
)
from .cycles import (
    steane_z_correct,
    steane_x_correct,
    iceberg_w_0_detect,
    iceberg_w_1_detect,
    iceberg_w_2_detect,
    iceberg_x_0_detect,
    iceberg_x_1_detect,
    iceberg_x_2_detect,
    iceberg_z_0_detect,
    iceberg_z_1_detect,
    iceberg_z_2_detect,
    x_dynamical_decoupling,
    x_transversal,
)
