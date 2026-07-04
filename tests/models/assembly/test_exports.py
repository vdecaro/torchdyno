def test_assembly_exports():
    import torchdyno.models.assembly as assembly

    for name in ["build_coupling", "FrequencyGate", "AdaDiagCore", "SCNCore"]:
        assert hasattr(assembly, name), name
