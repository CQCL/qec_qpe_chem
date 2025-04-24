# %%
import re
import itertools
import numpy as np
from pytket.circuit import (
    Qubit,
    Circuit,
    Bit,
    PauliExpBox,
    Pauli,
)
from h2xh2.encode.steane._steane import (
    _get_prep_non_ft,
    _get_prep_ft,
    _get_prep_ft_cond,
    _get_steane_decoder,
    _get_iceberg_zx,
    _get_iceberg_z,
    _get_iceberg_x,
    _get_iceberg_zx_cond,
    decoder,
    resolve_phase,
    ro_syndrome,
    ro_correction,
    get_prep_non_ft,
    get_measure,
    get_H,
    get_S,
    get_Sdg,
    get_Rz_direct,
    get_steane_correct_z,
    get_steane_correct_x,
    get_rz_non_ft,
    get_rz_meas_ft,
    get_rz_part_ft,
    get_rz_part_ft_goto,
    get_rzk_non_ft,
    get_rzk_meas_ft,
    get_rzk_part_ft,
    get_rzk_part_ft_goto,
    add_iceberg_x0,
    add_iceberg_w0,
    add_iceberg_z0,
    add_steane_z,
    add_x_transv,
    RzMode,
    RzOptionsBinFracPartFT,
    SteanePrimitives,
    RegNameMapping,
    SteaneCode,
)


def _run(circ: Circuit, n_shots: int = 1):
    # from pytket.extensions.qiskit import AerBackend
    from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline

    api_offline = QuantinuumAPIOffline()
    b = QuantinuumBackend(device_name="H2-1LE", api_handler=api_offline)
    c = b.get_compiled_circuit(circ, optimisation_level=0)
    r = b.run_circuit(c, n_shots=n_shots)
    return r


def test_pf_resolve_phase():
    """Resolve the phase into binary fraction."""
    n_bits = 5
    values = [
        [1.0, (1,)],
        [0.5, (0, 1)],
        [1.5, (1, 1)],
        [-1.5, (0, 1)],
    ]
    for phase, bits in values:
        tmp = resolve_phase(phase, n_bits)
        assert tuple(tmp) == bits


def test_pf_resolve_phase2():
    """Resolve the phase into binary fraction."""

    n_bits = 5
    for i in range(10):
        phase = 2 * np.random.random()

        tmp = resolve_phase(phase, n_bits)
        val = 0.0
        for i, b in enumerate(tmp):
            val += b * 2 ** (-i)
        diff = (val - phase) % 2.0
        if diff > 1.0:
            diff -= 2.0
        # print(val, phase, diff, 2 ** -n_bits)
        assert abs(diff) < 2**-n_bits


def test_pf_non_ft_prep():
    """Non-FT state preparation circuit."""
    qubits = [Qubit(i) for i in range(7)]
    circ = _get_prep_non_ft(qubits)
    circ.measure_all()
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=50)
    for k, _ in result.get_counts().items():
        parity = (-1) ** sum(k)
        assert parity == 1


def test_pf_non_ft_prep_x():
    """Non-FT state preparation followed by Logical X."""
    qubits = [Qubit(i) for i in range(7)]
    circ = _get_prep_non_ft(qubits)
    # logical X.
    for q in circ.qubits:
        circ.X(q)
    circ.measure_all()
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=50)
    for k, _ in result.get_counts().items():
        parity = (-1) ** sum(k)
        assert parity == -1


def test_pf_ftprep():
    """FT state preparation circuit (Goto state prep)."""
    qubits = [Qubit(i) for i in range(7)]
    bits = [Bit(i) for i in range(7)]
    gotoq = Qubit("gotoq", 0)
    gotob = Bit("gotob", 0)
    circ = _get_prep_ft(qubits, goto_qubit=gotoq, goto_bit=gotob)
    for q, b in zip(qubits, bits):
        circ.add_bit(b)
        circ.Measure(q, b)
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=50)
    # Check the parity.
    for k, _ in result.get_counts(cbits=bits).items():
        parity = (-1) ** sum(k)
        assert parity == 1
    # Check the goto bit.
    for k, _ in result.get_counts(cbits=[gotob]).items():
        assert k == (0,)


def test_pf_ftprep_cond():
    """Conditional FT state preparation circuit (Goto state prep)."""
    qubits = [Qubit(i) for i in range(7)]
    bits = [Bit(i) for i in range(7)]
    gotoq = Qubit("gotoq", 0)
    gotob = Bit("gotob", 0)
    condb = Bit("cond", 0)
    circ = Circuit()
    circ.add_bit(condb)
    circ.add_c_setbits([True], [condb])
    circ.append(
        _get_prep_ft_cond(qubits, goto_qubit=gotoq, goto_bit=gotob, condition=condb)
    )
    for q, b in zip(qubits, bits):
        circ.add_bit(b)
        circ.Measure(q, b)
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=50)
    for k, _ in result.get_counts(cbits=bits).items():
        parity = (-1) ** sum(k)
        assert parity == 1
    for k, _ in result.get_counts(cbits=[gotob]).items():
        assert k == (0,)


def test_pf_ftprep_cond_pass():
    """Conditional FT state preparation circuit (Goto state prep) to pass."""
    qubits = [Qubit(i) for i in range(7)]
    bits = [Bit(i) for i in range(7)]
    gotoq = Qubit("gotoq", 0)
    gotob = Bit("gotob", 0)
    condb = Bit("cond", 0)
    circ = Circuit()
    circ.add_bit(condb)
    circ.add_c_setbits([False], [condb])
    circ.append(
        _get_prep_ft_cond(qubits, goto_qubit=gotoq, goto_bit=gotob, condition=condb)
    )
    for q, b in zip(qubits, bits):
        circ.add_bit(b)
        circ.Measure(q, b)
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=50)
    for k, _ in result.get_counts(cbits=bits).items():
        assert k == (0, 0, 0, 0, 0, 0, 0)


def test_pf_steane_decoder():
    """Check the Steane decoder consistency."""
    ancl_bits = [Bit("ab", i) for i in range(7)]
    synd_bits = [Bit("sb", i) for i in range(3)]
    for syndrome in itertools.product(*[range(2)] * 3):
        qec_target = decoder(syndrome[::-1])
        circ = Circuit(7)
        if qec_target is not None:
            circ.X(qec_target)
        for q, b in zip(circ.qubits, ancl_bits):
            circ.add_bit(b)
            circ.Measure(q, b)
        circ.append(_get_steane_decoder(ancl_bits, synd_bits))
        # render_circuit_jupyter(circ)
        result = _run(circ)
        for k, _ in result.get_counts(cbits=synd_bits).items():
            assert k == syndrome[::-1]


def test_pf_steane_correct_z():
    """Steane EC for Z stabilizers (i.e., X errors)."""
    data_qubits = [Qubit("q", i) for i in range(7)]
    data_bits = [Bit("c", i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(7)]
    ancl_bits = [Bit("ab", i) for i in range(7)]
    synd_bits = [Bit("sb", i) for i in range(3)]
    goto_qubit = Qubit("xgq", 0)
    goto_bit = Bit("xgb", 0)
    for ii in range(7):
        circ = get_prep_non_ft(data_qubits)
        circ.add_barrier(data_qubits)
        circ.X(data_qubits[ii])
        circ.append(
            get_steane_correct_z(
                data_qubits,
                ancl_qubits,
                ancl_bits,
                synd_bits,
                goto_qubit=goto_qubit,
                goto_bit=goto_bit,
            )
        )
        circ.append(get_measure(data_qubits, data_bits))
        # render_circuit_jupyter(circ)
        result = _run(circ, n_shots=5)
        # Check the syndrome and corrected result.
        for k, _ in result.get_counts(cbits=data_bits + synd_bits).items():
            assert sum(k[7:]) > 0
            assert sum(ro_syndrome(k[:7])) == 0
        for k, _ in result.get_counts(cbits=[goto_bit]).items():
            assert k == (0,)


def test_pf_steane_correct_x():
    """Steane EC for X stabilizers (i.e., Z errors)."""
    data_qubits = [Qubit("q", i) for i in range(7)]
    data_bits = [Bit("c", i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(7)]
    ancl_bits = [Bit("ab", i) for i in range(7)]
    synd_bits = [Bit("sb", i) for i in range(3)]
    goto_qubit = Qubit("xgq", 0)
    goto_bit = Bit("xgb", 0)
    for ii in range(7):
        circ = get_prep_non_ft(data_qubits)
        circ.append(get_H(data_qubits))
        circ.add_barrier(data_qubits)
        circ.Z(data_qubits[ii])
        circ.append(
            get_steane_correct_x(
                data_qubits,
                ancl_qubits,
                ancl_bits,
                synd_bits,
                goto_qubit=goto_qubit,
                goto_bit=goto_bit,
            )
        )
        circ.append(get_measure(data_qubits, data_bits))
        # render_circuit_jupyter(circ)
        result = _run(circ, n_shots=5)
        # Check the syndrome and corrected result.
        for k, _ in result.get_counts(cbits=data_bits + synd_bits).items():
            assert sum(k[7:]) > 0
            assert sum(ro_syndrome(k[:7])) == 0
        for k, _ in result.get_counts(cbits=[goto_bit]).items():
            assert k == (0,)


def test_pf_iceberg_zx_zerr():
    data_qubits = [Qubit("q", i) for i in range(7)]
    data_bits = [Bit("c", i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(2)]
    ancl_bits = [Bit("ab", i) for i in range(2)]
    targets = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
    for ii in range(7):
        for isynd, synd in enumerate(targets):
            circ = get_prep_non_ft(data_qubits)
            circ.append(get_H(data_qubits))
            circ.add_barrier(data_qubits)
            circ.Z(data_qubits[ii])
            circ.append(
                _get_iceberg_zx(
                    isynd,
                    data_qubits,
                    ancl_qubits,
                    ancl_bits,
                )
            )
            circ.append(get_H(data_qubits))
            circ.append(get_measure(data_qubits, data_bits))
            # render_circuit_jupyter(circ)
            result = _run(circ, n_shots=5)
            for k, _ in result.get_counts(cbits=ancl_bits).items():
                if ii in synd:
                    assert k == (0, 1)
                else:
                    assert k == (0, 0)


def test_pf_iceberg_zx_xerr():
    data_qubits = [Qubit("q", i) for i in range(7)]
    data_bits = [Bit("c", i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(2)]
    ancl_bits = [Bit("ab", i) for i in range(2)]
    targets = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
    for ii in range(7):
        for isynd, synd in enumerate(targets):
            circ = get_prep_non_ft(data_qubits)
            circ.append(get_H(data_qubits))
            circ.add_barrier(data_qubits)
            circ.X(data_qubits[ii])
            circ.append(
                _get_iceberg_zx(
                    isynd,
                    data_qubits,
                    ancl_qubits,
                    ancl_bits,
                )
            )
            circ.append(get_measure(data_qubits, data_bits))
            # render_circuit_jupyter(circ)
            result = _run(circ, n_shots=5)
            for k, _ in result.get_counts(cbits=ancl_bits).items():
                if ii in synd:
                    assert k == (1, 0)
                else:
                    assert k == (0, 0)


def test_pf_iceberg_zx_xerr_pass():
    data_qubits = [Qubit("q", i) for i in range(7)]
    data_bits = [Bit("c", i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(2)]
    ancl_bits = [Bit("ab", i) for i in range(2)]
    condition = Bit("xc", 0)
    for ii in range(7):
        circ = get_prep_non_ft(data_qubits)
        circ.add_barrier(data_qubits)
        circ.X(data_qubits[ii])
        circ.append(
            _get_iceberg_zx_cond(
                0,
                data_qubits,
                ancl_qubits,
                ancl_bits,
                condition,
            )
        )
        circ.append(get_measure(data_qubits, data_bits))
        # render_circuit_jupyter(circ)
        result = _run(circ, n_shots=5)
        for k, _ in result.get_counts(cbits=ancl_bits).items():
            # print(k)
            assert k == (0, 0)
        for k, _ in result.get_counts(cbits=data_bits).items():
            # print(k, ro_syndrome(k))
            assert sum(ro_syndrome(k)) > 0


def test_pf_iceberg_z():
    data_qubits = [Qubit("q", i) for i in range(7)]
    data_bits = [Bit("c", i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(2)]
    ancl_bits = [Bit("ab", i) for i in range(2)]
    targets = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
    for ii in range(7):
        for isynd, synd in enumerate(targets):
            circ = get_prep_non_ft(data_qubits)
            circ.add_barrier(data_qubits)
            circ.X(data_qubits[ii])
            circ.append(
                _get_iceberg_z(
                    isynd,
                    data_qubits,
                    ancl_qubits,
                    ancl_bits,
                )
            )
            circ.add_barrier(data_qubits)
            circ.append(get_measure(data_qubits, data_bits))
            # render_circuit_jupyter(circ)
            result = _run(circ, n_shots=5)
            for k, _ in result.get_counts(cbits=ancl_bits).items():
                if ii in synd:
                    assert k == (1, 0)
                else:
                    assert k == (0, 0)


def test_pf_iceberg_x():
    data_qubits = [Qubit("q", i) for i in range(7)]
    data_bits = [Bit("c", i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(2)]
    ancl_bits = [Bit("ab", i) for i in range(2)]
    targets = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
    for ii in range(7):
        for isynd, synd in enumerate(targets):
            circ = get_prep_non_ft(data_qubits)
            circ.append(get_H(data_qubits))
            circ.add_barrier(data_qubits)
            circ.Z(data_qubits[ii])
            circ.append(
                _get_iceberg_z(
                    isynd,
                    data_qubits,
                    ancl_qubits,
                    ancl_bits,
                )
            )
            circ.add_barrier(data_qubits)
            circ.append(get_H(data_qubits))
            circ.append(get_measure(data_qubits, data_bits))
            # render_circuit_jupyter(circ)
            result = _run(circ, n_shots=10)
            for k, _ in result.get_counts(cbits=ancl_bits).items():
                if ii in synd:
                    assert k == (1, 0)
                else:
                    assert k == (0, 0)


def test_pf_iceberg_x():
    data_qubits = [Qubit("q", i) for i in range(7)]
    data_bits = [Bit("c", i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(2)]
    ancl_bits = [Bit("ab", i) for i in range(2)]
    targets = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
    for ii in range(7):
        for isynd, synd in enumerate(targets):
            circ = get_prep_non_ft(data_qubits)
            circ.add_barrier(data_qubits)
            circ.append(get_H(data_qubits))
            circ.Z(data_qubits[ii])
            circ.append(
                _get_iceberg_x(
                    isynd,
                    data_qubits,
                    ancl_qubits,
                    ancl_bits,
                )
            )
            circ.add_barrier(data_qubits)
            circ.append(get_measure(data_qubits, data_bits))
            # render_circuit_jupyter(circ)
            result = _run(circ, n_shots=10)
            for k, _ in result.get_counts(cbits=ancl_bits).items():
                if ii in synd:
                    assert k == (1, 0)
                else:
                    assert k == (0, 0)


def test_pf_rz_nonft_t():
    """Non-FT T gate operation by the gate teleportation."""
    data_qubits = [Qubit(i) for i in range(7)]
    data_bits = [Bit(i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(7)]
    ancl_bits = [Bit("xc", i) for i in range(3)]
    cond_bit = Bit("xCond", 0)
    # Non-FT plust state preparation.
    circ = get_prep_non_ft(data_qubits)
    circ.append(get_H(data_qubits))
    # Non-FT magic state injection.
    phase = 0.25
    circ.append(get_rz_non_ft(phase, data_qubits, ancl_qubits, ancl_bits, cond_bit))
    circ.append(get_S(data_qubits, condition=cond_bit))
    # Non-FT direct Rz operation to cancel the Rz just applied.
    circ.append(get_Rz_direct(-phase, data_qubits))
    # X measurement.
    circ.append(get_H(data_qubits))
    circ.append(get_measure(data_qubits, data_bits))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    for k, _ in result.get_counts(cbits=data_bits).items():
        parity = (-1) ** sum(k)
        # print(k, parity)
        assert parity == 1


def test_pf_rz_nonft_tdg():
    """Non-FT Tdg gate operation by the gate teleportation."""
    data_qubits = [Qubit(i) for i in range(7)]
    data_bits = [Bit(i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(7)]
    ancl_bits = [Bit("xc", i) for i in range(3)]
    cond_bit = Bit("xCond", 0)
    # Non-FT plust state preparation.
    circ = get_prep_non_ft(data_qubits)
    circ.append(get_H(data_qubits))
    # Non-FT magic state injection.
    phase = -0.25
    circ.append(get_rz_non_ft(phase, data_qubits, ancl_qubits, ancl_bits, cond_bit))
    circ.append(get_Sdg(data_qubits, condition=cond_bit))
    # Non-FT direct Rz operation to cancel the Rz just applied.
    circ.append(get_Rz_direct(-phase, data_qubits))
    # X measurement.
    circ.append(get_H(data_qubits))
    circ.append(get_measure(data_qubits, data_bits))
    # render_circuit_jupyter(circ)
    # return
    result = _run(circ, n_shots=10)
    for k, _ in result.get_counts(cbits=data_bits).items():
        parity = (-1) ** sum(k)
        # print(k, parity)
        assert parity == 1


def test_pf_rz_ft_meas():
    """FT-measurement T gate operation by the gate teleportation."""
    data_qubits = [Qubit(i) for i in range(7)]
    data_bits = [Bit(i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(7)]
    ancl_bits = [Bit("xc", i) for i in range(7 + 4)]
    cond_bit = Bit("xCond", 0)
    # Non-FT plust state preparation.
    circ = get_prep_non_ft(data_qubits)
    circ.append(get_H(data_qubits))
    circ.add_bit(cond_bit)
    circ.add_c_setbits([True], [cond_bit])
    # Non-FT magic state injection.
    phase = 0.25
    circ.append(get_rz_meas_ft(phase, data_qubits, ancl_qubits, ancl_bits, cond_bit))
    circ.append(get_S(data_qubits, condition=cond_bit))
    # Non-FT direct Rz operation to cancel the Rz just applied.
    circ.append(get_Rz_direct(-phase, data_qubits))
    # X measurement.
    circ.append(get_H(data_qubits))
    circ.append(get_measure(data_qubits, data_bits))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    for k, _ in result.get_counts(cbits=data_bits).items():
        parity = (-1) ** sum(k)
        # print(k, parity)
        assert parity == 1


def test_pf_rz_meas_ft_1q_error():
    """FT Measurement T with an 1Q error."""
    data_qubits = [Qubit(i) for i in range(7)]
    data_bits = [Bit(i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(7)]
    ancl_bits = [Bit("xc", i) for i in range(7 + 4)]
    cond_bit = Bit("xCond", 0)
    # Non-FT plust state preparation.
    circ = get_prep_non_ft(data_qubits)
    circ.append(get_H(data_qubits))
    circ.add_bit(cond_bit)
    circ.add_c_setbits([True], [cond_bit])
    # Add one qubit error.
    circ.add_barrier(circ.qubits)
    circ.X(data_qubits[1])
    circ.add_barrier(circ.qubits)
    # Non-FT magic state injection.
    phase = 0.25
    circ.append(get_rz_meas_ft(phase, data_qubits, ancl_qubits, ancl_bits, cond_bit))
    circ.append(get_S(data_qubits, condition=cond_bit))
    # Cancel the error (mimicking Error correction).
    circ.add_barrier(circ.qubits)
    circ.X(data_qubits[1])
    circ.add_barrier(circ.qubits)
    # Non-FT direct Rz operation to cancel the Rz just applied.
    circ.append(get_Rz_direct(-phase, data_qubits))
    # X measurement.
    circ.append(get_H(data_qubits))
    circ.append(get_measure(data_qubits, data_bits))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=20)
    for k, _ in result.get_counts(cbits=data_bits).items():
        k_corr = ro_correction(k)
        parity = (-1) ** sum(k)
        # print("raw", k, parity, ro_syndrome(k))
        k_corr = ro_correction(k)
        parity = (-1) ** sum(k_corr)
        # print("cor", k_corr, parity)
        # print()
        assert parity == 1


def test_pf_rz_part_ft():
    """Partially FT T gate operation by the gate teleportation."""
    data_qubits = [Qubit(i) for i in range(7)]
    data_bits = [Bit(i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(9)]
    ancl_bits = [Bit("xc", i) for i in range(7)]
    synd_bits = [Bit("xs", i) for i in range(4)]
    flag_bit = Bit("xFlag", 0)
    cond_bit = Bit("xCond", 0)
    # Non-FT plust state preparation.
    circ = get_prep_non_ft(data_qubits)
    circ.append(get_H(data_qubits))
    circ.add_bit(cond_bit)
    circ.add_c_setbits([True], [cond_bit])
    # Non-FT magic state injection.
    phase = 0.25
    circ.append(
        get_rz_part_ft(
            phase,
            data_qubits,
            ancl_qubits,
            ancl_bits,
            synd_bits,
            flag_bit,
            n_rus=1,
            condition=cond_bit,
        )
    )
    # circ.append(get_rz_meas_ft(phase, data_qubits, ancl_qubits, ancl_bits, cond_bit))
    circ.append(get_S(data_qubits, condition=cond_bit))
    # Non-FT direct Rz operation to cancel the Rz just applied.
    circ.append(get_Rz_direct(-phase, data_qubits))
    # X measurement.
    circ.append(get_H(data_qubits))
    circ.append(get_measure(data_qubits, data_bits))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    for k, _ in result.get_counts(cbits=data_bits).items():
        parity = (-1) ** sum(k)
        # print(k, parity)
        assert parity == 1


def test_pf_rz_part_ft_goto():
    """Partially FT T gate operation by the gate teleportation."""
    data_qubits = [Qubit(i) for i in range(7)]
    data_bits = [Bit(i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(9)]
    ancl_bits = [Bit("xc", i) for i in range(7)]
    synd_bits = [Bit("xs", i) for i in range(5)]
    flag_bit = Bit("xFlag", 0)
    cond_bit = Bit("xCond", 0)
    # Non-FT plust state preparation.
    circ = get_prep_non_ft(data_qubits)
    circ.append(get_H(data_qubits))
    circ.add_bit(cond_bit)
    circ.add_c_setbits([True], [cond_bit])
    # Non-FT magic state injection.
    phase = 0.25
    circ.append(
        get_rz_part_ft_goto(
            phase,
            data_qubits,
            ancl_qubits,
            ancl_bits,
            synd_bits,
            flag_bit,
            n_rus=2,
            condition=cond_bit,
        )
    )
    # circ.append(get_rz_meas_ft(phase, data_qubits, ancl_qubits, ancl_bits, cond_bit))
    circ.append(get_S(data_qubits, condition=cond_bit))
    # Non-FT direct Rz operation to cancel the Rz just applied.
    circ.append(get_Rz_direct(-phase, data_qubits))
    # X measurement.
    circ.append(get_H(data_qubits))
    circ.append(get_measure(data_qubits, data_bits))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    for k, _ in result.get_counts(cbits=data_bits).items():
        parity = (-1) ** sum(k)
        # print(k, parity)
        assert parity == 1


def test_pf_rz_nonft_err():
    """Non-FT T gate operation encoutering an error."""
    data_qubits = [Qubit(i) for i in range(7)]
    data_bits = [Bit(i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(7)]
    ancl_bits = [Bit("xc", i) for i in range(3)]
    cond_bit = Bit("xCond", 0)
    # Non-FT plust state preparation.
    circ = get_prep_non_ft(data_qubits)
    circ.append(get_H(data_qubits))
    circ.add_bit(cond_bit)
    circ.add_c_setbits([True], [cond_bit])
    # Add one qubit error.
    circ.add_barrier(circ.qubits)
    circ.X(data_qubits[1])
    circ.add_barrier(circ.qubits)
    # Non-FT magic state injection.
    phase = 0.25
    circ.append(get_rz_non_ft(phase, data_qubits, ancl_qubits, ancl_bits, cond_bit))
    circ.append(get_S(data_qubits, condition=cond_bit))
    # Cancel the error (mimicking Error correction).
    circ.add_barrier(circ.qubits)
    circ.X(data_qubits[1])
    circ.add_barrier(circ.qubits)
    # Non-FT direct Rz operation to cancel the Rz just applied.
    circ.append(get_Rz_direct(-phase, data_qubits))
    # X measurement.
    circ.append(get_H(data_qubits))
    circ.append(get_measure(data_qubits, data_bits))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=20)
    ls_parity_checks: list[bool] = []
    for k, _ in result.get_counts(cbits=data_bits).items():
        k_corr = ro_correction(k)
        parity = (-1) ** sum(k)
        # print("raw", k, parity, ro_syndrome(k))
        k_corr = ro_correction(k)
        parity = (-1) ** sum(k_corr)
        # print("cor", k_corr, parity)
        # print()
        ls_parity_checks.append(parity == 1)
    assert not all(ls_parity_checks)


def test_pf_rzk_nonft():
    """Non-FT Rz(x/2^k) operation with recursive gate teleportation."""
    data_qubits = [Qubit(i) for i in range(7)]
    data_bits = [Bit(i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(7)]
    ancl_bits = [Bit("xc", i) for i in range(3)]
    condition = Bit("xf", 0)
    # Non-FT plust state preparation.
    circ = Circuit()
    circ.append(get_prep_non_ft(data_qubits))
    circ.append(get_H(data_qubits))
    # Non-FT magic state injection.
    angle_bits = [1, 1, 0, 1, 0, 1]
    circ.append(
        get_rzk_non_ft(angle_bits, data_qubits, ancl_qubits, ancl_bits, condition)
    )
    # Non-FT direct Rz operation to cancel the Rz just applied.
    phase = sum([kval * 2**-i for i, kval in enumerate(angle_bits)])
    # print(phase, [kval * 2 ** -i for i, kval in enumerate(angle_bits)])
    circ.add_barrier(data_qubits)
    circ.append(get_Rz_direct(-phase, data_qubits))
    # X measurement.
    circ.append(get_H(data_qubits))
    circ.append(get_measure(data_qubits, data_bits))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    for k, _ in result.get_counts(cbits=data_bits).items():
        parity = (-1) ** sum(k)
        assert parity == 1


def test_pf_rzk_meas_ft():
    """FT-measurement Rz(x/2^k) operation with recursive gate teleportation."""
    data_qubits = [Qubit(i) for i in range(7)]
    data_bits = [Bit(i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(7)]
    ancl_bits = [Bit("xc", i) for i in range(7 + 4)]
    condition = Bit("xf", 0)
    # Non-FT plust state preparation.
    circ = Circuit()
    circ.append(get_prep_non_ft(data_qubits))
    circ.append(get_H(data_qubits))
    # Non-FT magic state injection.
    angle_bits = [1, 1, 0, 1, 0, 1]
    circ.append(
        get_rzk_meas_ft(angle_bits, data_qubits, ancl_qubits, ancl_bits, condition)
    )
    # Non-FT direct Rz operation to cancel the Rz just applied.
    phase = sum([kval * 2**-i for i, kval in enumerate(angle_bits)])
    # print(phase, [kval * 2 ** -i for i, kval in enumerate(angle_bits)])
    circ.add_barrier(data_qubits)
    circ.append(get_Rz_direct(-phase, data_qubits))
    # X measurement.
    circ.append(get_H(data_qubits))
    circ.append(get_measure(data_qubits, data_bits))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    for k, _ in result.get_counts(cbits=data_bits).items():
        parity = (-1) ** sum(k)
        assert parity == 1


def test_pf_rzk_part_ft():
    """Partially FT Rz(x/2^k) operation with recursive gate teleportation."""
    data_qubits = [Qubit(i) for i in range(7)]
    data_bits = [Bit(i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(9)]
    ancl_bits = [Bit("xc", i) for i in range(13)]
    condition = Bit("xf", 0)
    disc_bit = Bit("xDiscard", 0)
    # Non-FT plust state preparation.
    circ = Circuit()
    circ.append(get_prep_non_ft(data_qubits))
    circ.append(get_H(data_qubits))
    # Non-FT magic state injection.
    # angle_bits = [1, 1, 0, 1, 0, 1]
    angle_bits = [0, 0, 1, 0, 0, 0]
    n_rus = 2
    circ.append(
        get_rzk_part_ft(
            angle_bits,
            data_qubits,
            ancl_qubits,
            ancl_bits,
            disc_bit,
            condition,
            n_rus,
        )
    )
    # Non-FT direct Rz operation to cancel the Rz just applied.
    phase = sum([kval * 2**-i for i, kval in enumerate(angle_bits)])
    # print(phase, [kval * 2 ** -i for i, kval in enumerate(angle_bits)])
    circ.add_barrier(data_qubits)
    circ.append(get_Rz_direct(-phase, data_qubits))
    # X measurement.
    circ.append(get_H(data_qubits))
    circ.append(get_measure(data_qubits, data_bits))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    for k, _ in result.get_counts(cbits=data_bits).items():
        parity = (-1) ** sum(k)
        assert parity == 1
    for k, _ in result.get_counts(cbits=[disc_bit]).items():
        # print(k)
        assert k == (0,)


def test_pf_rzk_part_ft_goto():
    """Partially FT Rz(x/2^k) operation with recursive gate teleportation."""
    data_qubits = [Qubit(i) for i in range(7)]
    data_bits = [Bit(i) for i in range(7)]
    ancl_qubits = [Qubit("xq", i) for i in range(9)]
    ancl_bits = [Bit("xc", i) for i in range(13)]
    condition = Bit("xf", 0)
    disc_bit = Bit("xDiscard", 0)
    # Non-FT plust state preparation.
    circ = Circuit()
    circ.append(get_prep_non_ft(data_qubits))
    circ.append(get_H(data_qubits))
    # Non-FT magic state injection.
    angle_bits = [1, 1, 0, 1, 0, 1]
    # angle_bits = [0, 0, 1, 0, 0, 0]
    n_rus = 2
    circ.append(
        get_rzk_part_ft_goto(
            angle_bits,
            data_qubits,
            ancl_qubits,
            ancl_bits,
            disc_bit,
            condition,
            n_rus,
        )
    )
    # Non-FT direct Rz operation to cancel the Rz just applied.
    phase = sum([kval * 2**-i for i, kval in enumerate(angle_bits)])
    # print(phase, [kval * 2 ** -i for i, kval in enumerate(angle_bits)])
    circ.add_barrier(data_qubits)
    circ.append(get_Rz_direct(-phase, data_qubits))
    # X measurement.
    circ.append(get_H(data_qubits))
    circ.append(get_measure(data_qubits, data_bits))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    for k, _ in result.get_counts(cbits=data_bits).items():
        parity = (-1) ** sum(k)
        assert parity == 1
    for k, _ in result.get_counts(cbits=[disc_bit]).items():
        # print(k)
        assert k == (0,)


def test_prim_x():
    """Test `SteanePrimitives.X`."""
    sp = SteanePrimitives(1, 1)
    circ = sp.prep_non_ft()
    circ.append(sp.X(0))
    circ.append(sp.Measure(0, 0))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=20)
    for k, _ in result.get_counts(cbits=sp.data_bits).items():
        parity = (-1) ** sum(k)
        # print(k, parity)
        assert parity == -1


def test_prim_get_steane_x_nonft():
    sp = SteanePrimitives(1, 1)
    circ = sp.prep_non_ft()
    circ.append(sp.H(0))
    circ.add_barrier(sp.data_qubits)
    circ.Z(0)
    circ.append(sp.steane_correct_x_non_ft(0))
    circ.append(sp.H(0))
    circ.append(sp.Measure(0, 0))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    # Check if the error correction works.
    for k, _ in result.get_counts(cbits=sp.data_bits).items():
        parity = (-1) ** sum(k)
        # print(k, parity)
        assert parity == 1
    rnm = RegNameMapping()
    bits = [b for b in result.get_bitlist() if re.match(rnm.synd_bits, b.reg_name)]
    # Check if the non-trivial syndrome is observed.
    for k, _ in result.get_counts(cbits=bits).items():
        # print(k)
        assert sum(k) > 0


def test_prim_get_steane_z_nonft():
    sp = SteanePrimitives(1, 1)
    circ = sp.prep_non_ft()
    circ.add_barrier(sp.data_qubits)
    circ.X(0)
    circ.append(sp.steane_correct_z_non_ft(0))
    circ.append(sp.Measure(0, 0))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    # Check if the error correction works.
    for k, _ in result.get_counts(cbits=sp.data_bits).items():
        parity = (-1) ** sum(k)
        # print(k, parity)
        assert parity == 1
    rnm = RegNameMapping()
    bits = [b for b in result.get_bitlist() if re.match(rnm.synd_bits, b.reg_name)]
    # Check if the non-trivial syndrome is observed.
    for k, _ in result.get_counts(cbits=bits).items():
        # print(k)
        assert sum(k) > 0


def test_prim_get_steane_x_nonft():
    sp = SteanePrimitives(1, 1)
    circ = sp.prep_non_ft()
    circ.append(sp.H(0))
    circ.add_barrier(sp.data_qubits)
    circ.Z(0)
    circ.append(sp.steane_correct_x(0))
    circ.append(sp.H(0))
    circ.append(sp.Measure(0, 0))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    # Check if the error correction works.
    for k, _ in result.get_counts(cbits=sp.data_bits).items():
        parity = (-1) ** sum(k)
        # print(k, parity)
        assert parity == 1
    rnm = RegNameMapping()
    bits = [b for b in result.get_bitlist() if re.match(rnm.synd_bits, b.reg_name)]
    # Check if the non-trivial syndrome is observed.
    for k, _ in result.get_counts(cbits=bits).items():
        # print(k)
        assert sum(k) > 0
    # CHeck the discard_bit.
    for k, _ in result.get_counts(cbits=[sp.discard_bit]).items():
        # print(k)
        assert k == (0,)


def test_prim_get_steane_x_rus():
    """Steane X syndrome with RUS state prep."""
    sp = SteanePrimitives(1, 1)
    circ = sp.prep_non_ft()
    circ.append(sp.H(0))
    circ.add_barrier(sp.data_qubits)
    circ.Z(0)
    circ.append(sp.steane_correct_x(0, n_rus=2))
    circ.append(sp.H(0))
    circ.append(sp.Measure(0, 0))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    # Check if the error correction works.
    for k, _ in result.get_counts(cbits=sp.data_bits).items():
        parity = (-1) ** sum(k)
        # print(k, parity)
        assert parity == 1
    rnm = RegNameMapping()
    bits = [b for b in result.get_bitlist() if re.match(rnm.synd_bits, b.reg_name)]
    # Check if the non-trivial syndrome is observed.
    for k, _ in result.get_counts(cbits=bits).items():
        # print(k)
        assert sum(k) > 0
    # CHeck the discard_bit.
    for k, _ in result.get_counts(cbits=[sp.discard_bit]).items():
        # print(k)
        assert k == (0,)


def test_prim_get_steane_z_rus():
    """Steane Z syndrome with RUS state prep."""
    sp = SteanePrimitives(1, 1)
    circ = sp.prep_non_ft()
    circ.add_barrier(sp.data_qubits)
    circ.X(0)
    circ.append(sp.steane_correct_z(0, n_rus=2))
    circ.append(sp.Measure(0, 0))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    # Check if the error correction works.
    for k, _ in result.get_counts(cbits=sp.data_bits).items():
        parity = (-1) ** sum(k)
        # print(k, parity)
        assert parity == 1
    rnm = RegNameMapping()
    bits = [b for b in result.get_bitlist() if re.match(rnm.synd_bits, b.reg_name)]
    # Check if the non-trivial syndrome is observed.
    for k, _ in result.get_counts(cbits=bits).items():
        # print(k)
        assert sum(k) > 0
    # CHeck the discard_bit.
    for k, _ in result.get_counts(cbits=[sp.discard_bit]).items():
        # print(k)
        assert k == (0,)


def test_prim_get_iceberg_zx_xerr():
    """Iceberg XZ syndrome through `SteanePrimitives`"""
    sp = SteanePrimitives(1, 1)
    circ = sp.prep_non_ft()
    circ.add_barrier(sp.data_qubits)
    circ.X(0)
    circ.append(sp.iceberg_detect_zx(0, 0))
    circ.append(sp.Measure(0, 0))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    # Check if the error correction works.
    has_error = False
    for k, _ in result.get_counts(cbits=sp.data_bits).items():
        parity = (-1) ** sum(k)
        has_error = has_error or parity != 1
        # print(k, parity)
    assert has_error
    # Check the discard_bit.
    for k, _ in result.get_counts(cbits=sp.ancila_bits[:2]).items():
        # print(k)
        assert k == (1, 0)
    for k, _ in result.get_counts(cbits=[sp.discard_bit]).items():
        # print(k)
        assert k == (1,)


def test_prim_get_iceberg_zx_zerr():
    """Iceberg XZ syndrome through `SteanePrimitives`"""
    sp = SteanePrimitives(1, 1)
    circ = sp.prep_non_ft()
    circ.add_barrier(sp.data_qubits)
    circ.append(sp.H(0))
    circ.Z(0)
    circ.append(sp.iceberg_detect_zx(0, 0))
    circ.append(sp.H(0))
    circ.append(sp.Measure(0, 0))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    # Check if the error correction works.
    has_error = False
    for k, _ in result.get_counts(cbits=sp.data_bits).items():
        parity = (-1) ** sum(k)
        has_error = has_error or parity != 1
        # print(k, parity)
    assert has_error
    # Check the discard_bit.
    for k, _ in result.get_counts(cbits=sp.ancila_bits[:2]).items():
        # print(k)
        assert k == (0, 1)
    for k, _ in result.get_counts(cbits=[sp.discard_bit]).items():
        # print(k)
        assert k == (1,)


def test_prim_get_iceberg_z():
    """Iceberg Z syndrome through `SteanePrimitives`"""
    sp = SteanePrimitives(1, 1)
    circ = sp.prep_non_ft()
    circ.add_barrier(sp.data_qubits)
    circ.X(0)
    circ.append(sp.iceberg_detect_z(0, 0))
    circ.append(sp.Measure(0, 0))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    # Check if the error correction works.
    has_error = False
    for k, _ in result.get_counts(cbits=sp.data_bits).items():
        parity = (-1) ** sum(k)
        has_error = has_error or parity != 1
        # print(k, parity)
    assert has_error
    # Check the discard_bit.
    for k, _ in result.get_counts(cbits=sp.ancila_bits[:2]).items():
        # print(k)
        assert k == (1, 0)
    for k, _ in result.get_counts(cbits=[sp.discard_bit]).items():
        # print(k)
        assert k == (1,)


def test_prim_get_iceberg_x():
    """Iceberg X syndrome through `SteanePrimitives`"""
    sp = SteanePrimitives(1, 1)
    circ = sp.prep_non_ft()
    circ.add_barrier(sp.data_qubits)
    circ.append(sp.H(0))
    circ.Z(0)
    circ.append(sp.iceberg_detect_x(0, 0))
    circ.append(sp.H(0))
    circ.append(sp.Measure(0, 0))
    # render_circuit_jupyter(circ)
    result = _run(circ, n_shots=10)
    # Check if the error correction works.
    has_error = False
    for k, _ in result.get_counts(cbits=sp.data_bits).items():
        parity = (-1) ** sum(k)
        has_error = has_error or parity != 1
        # print(k, parity)
    assert has_error
    # Check the discard_bit.
    for k, _ in result.get_counts(cbits=sp.ancila_bits[:2]).items():
        # print(k)
        assert k == (1, 0)
    for k, _ in result.get_counts(cbits=[sp.discard_bit]).items():
        # print(k)
        assert k == (1,)


def test_sc_spam_nonft():
    sc = SteaneCode()
    circ = Circuit(2, 2)
    circ.X(1)
    circ.measure_all()
    circe = sc.get_encoded_circuit(circ)
    # render_circuit_jupyter(circ)
    # render_circuit_jupyter(circe)
    resulte = _run(circe)
    result = sc.get_decoded_result(resulte)
    for k, _ in result.get_counts().items():
        # print(k)
        assert k == (0, 1)


def test_sc_spam_nonft_hh():
    sc = SteaneCode()
    circ = Circuit(1, 1)
    circ.add_barrier(circ.qubits)
    circ.H(0)
    circ.add_barrier(circ.qubits)
    circ.H(0)
    circ.measure_all()
    circe = sc.get_encoded_circuit(circ)
    # render_circuit_jupyter(circ)
    # render_circuit_jupyter(circe)
    resulte = _run(circe)
    result = sc.get_decoded_result(resulte)
    for k, _ in result.get_counts().items():
        # print(k)
        assert k == (0,)


def test_sc_spam_pft_ttdg():
    sc = SteaneCode()
    circ = Circuit(1, 1)
    n_bits = 5
    phase = 3 * 2 ** -(n_bits - 1)
    # print(phase)
    circ.Rz(phase, 0)
    circ.add_pauliexpbox(
        PauliExpBox([Pauli.Z], -phase),
        [0],
    )
    circ.add_barrier(circ.qubits)
    # add_steane_z(circ, 0)
    circ.measure_all()
    circe = sc.get_encoded_circuit(
        circ,
        rz_mode=RzMode.BIN_FRAC_PART_FT,
        rz_options=RzOptionsBinFracPartFT(max_bits=5),
    )
    circ.remove_blank_wires()
    resulte = _run(circe, n_shots=10)
    result = sc.get_decoded_result(resulte)
    for k, _ in result.get_counts().items():
        # print(k)
        assert k == (0,)


def test_sc_spam_pft_ttdg_goto():
    sc = SteaneCode()
    circ = Circuit(1, 1)
    n_bits = 5
    phase = 3 * 2 ** -(n_bits - 1)
    # print(phase)
    circ.Rz(phase, 0)
    circ.add_pauliexpbox(
        PauliExpBox([Pauli.Z], -phase),
        [0],
    )
    circ.add_barrier(circ.qubits)
    # add_steane_z(circ, 0)
    circ.measure_all()
    circe = sc.get_encoded_circuit(
        circ,
        rz_mode=RzMode.BIN_FRAC_PART_FT_GOTO,
        rz_options=RzOptionsBinFracPartFT(max_bits=5),
    )
    circ.remove_blank_wires()
    resulte = _run(circe, n_shots=10)
    result = sc.get_decoded_result(resulte)
    for k, _ in result.get_counts().items():
        # print(k)
        assert k == (0,)


def test_sc_spam_nonft_ttdg():
    sc = SteaneCode()
    circ = Circuit(1, 1)
    n_bits = 5
    phase = 3 * 2 ** -(n_bits - 1)
    # print(phase)
    circ.Rz(phase, 0)
    circ.add_barrier(circ.qubits)
    # circ.Tdg(0)
    add_steane_z(circ, 0)
    circ.measure_all()
    circe = sc.get_encoded_circuit(circ, rz_mode=RzMode.BIN_FRAC_NON_FT)
    # render_circuit_jupyter(circ)
    # render_circuit_jupyter(circe)
    resulte = _run(circe)
    result = sc.get_decoded_result(resulte)
    for k, _ in result.get_counts().items():
        # print(k)
        assert k == (0,)


def test_sc_discard():
    sc = SteaneCode()
    circ = Circuit(1, 1)
    add_iceberg_x0(circ, 0)
    circ.measure_all()
    circe = sc.get_encoded_circuit(circ)
    # Force to turn the discard bit to "1".
    circe.add_c_setbits([True], [sc._primitives.discard_bit])
    # render_circuit_jupyter(circ)
    # render_circuit_jupyter(circe)
    resulte = _run(circe)
    result = sc.get_decoded_result(resulte)
    assert len(result.get_counts()) == 0


def test_sc_add_transv_x():
    sc = SteaneCode()
    circ = Circuit(2)
    add_x_transv(circ)
    circ.measure_all()
    circe = sc.get_encoded_circuit(circ)
    # render_circuit_jupyter(circ)
    # render_circuit_jupyter(circe)
    resulte = _run(circe)
    result = sc.get_decoded_result(resulte)
    for k, _ in result.get_counts().items():
        # print(k)
        assert k == (1, 1)


def test_sc_add_iceberg_zx0():
    sc = SteaneCode()
    circ = Circuit(1)
    add_iceberg_w0(circ, 0)
    circ.measure_all()
    circe = sc.get_encoded_circuit(circ)
    # render_circuit_jupyter(circ)
    # render_circuit_jupyter(circe)
    resulte = _run(circe)
    result = sc.get_decoded_result(resulte)
    for k, _ in result.get_counts().items():
        # print(k)
        assert k == (0,)


def test_sc_add_iceberg_z0():
    sc = SteaneCode()
    circ = Circuit(1)
    add_iceberg_z0(circ, 0)
    circ.measure_all()
    circe = sc.get_encoded_circuit(circ)
    # render_circuit_jupyter(circ)
    # render_circuit_jupyter(circe)
    resulte = _run(circe)
    result = sc.get_decoded_result(resulte)
    for k, _ in result.get_counts().items():
        # print(k)
        assert k == (0,)


def test_sc_add_iceberg_x0():
    sc = SteaneCode()
    circ = Circuit(1)
    add_iceberg_x0(circ, 0)
    circ.measure_all()
    circe = sc.get_encoded_circuit(circ)
    # render_circuit_jupyter(circ)
    # render_circuit_jupyter(circe)
    resulte = _run(circe)
    result = sc.get_decoded_result(resulte)
    for k, _ in result.get_counts().items():
        # print(k)
        assert k == (0,)


def test_pf_resolve_phase():
    n_bits = 5
    values = [
        [1.0, (1, 0, 0, 0, 0)],
        [0.5, (0, 1, 0, 0, 0)],
        [1.5, (1, 1, 0, 0, 0)],
        [-1.5, (0, 1, 0, 0, 0)],
    ]
    for phase, bits in values:
        tmp = resolve_phase(phase, n_bits)
        # print(tmp)
        assert tuple(tmp) == bits


def test_pf_resolve_phase2():
    import numpy as np

    n_bits = 5
    for i in range(10):
        phase = 2 * np.random.random()

        tmp = resolve_phase(phase, n_bits)
        val = 0.0
        for i, b in enumerate(tmp):
            val += b * 2 ** (-i)
        diff = (val - phase) % 2.0
        if diff > 1.0:
            diff -= 2.0
        # print(val, phase, diff, 2 ** -n_bits)
        assert abs(diff) < 2**-n_bits


if __name__ == "__main__":
    from pytket.circuit.display import render_circuit_jupyter

    # Primitive functions tests.
    test_pf_resolve_phase()
    test_pf_resolve_phase2()
    test_pf_non_ft_prep()
    test_pf_non_ft_prep_x()
    test_pf_ftprep()
    test_pf_ftprep_cond()
    test_pf_ftprep_cond_pass()
    test_pf_steane_decoder()
    test_pf_steane_correct_z()
    test_pf_steane_correct_x()
    test_pf_iceberg_zx_zerr()
    test_pf_iceberg_zx_xerr()
    test_pf_iceberg_z()
    test_pf_iceberg_x()
    test_pf_rz_nonft_t()
    test_pf_rz_nonft_tdg()
    test_pf_rz_ft_meas()
    test_pf_rz_part_ft()
    test_pf_rz_part_ft_goto()
    test_pf_rz_meas_ft_1q_error()
    test_pf_rz_nonft_err()
    test_pf_rzk_nonft()
    test_pf_rzk_meas_ft()
    test_pf_rzk_part_ft()
    test_pf_rzk_part_ft_goto()
    test_prim_x()

    # SteanePrimitives tests.
    test_prim_get_steane_x_nonft()
    test_prim_get_steane_z_nonft()
    test_prim_get_steane_x_rus()
    test_prim_get_steane_z_rus()
    test_prim_get_iceberg_x()
    test_prim_get_iceberg_z()
    test_prim_get_iceberg_zx_xerr()
    test_prim_get_iceberg_zx_zerr()

    # SteaneCode tests.
    test_sc_spam_nonft()
    test_sc_discard()
    test_sc_spam_nonft_hh()
    test_sc_spam_nonft_ttdg()
    test_sc_spam_pft_ttdg()
    test_sc_spam_pft_ttdg_goto()
    test_sc_add_transv_x()
    test_sc_add_iceberg_zx0()
    test_sc_add_iceberg_z0()
    test_sc_add_iceberg_x0()
