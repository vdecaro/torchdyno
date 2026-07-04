def test_training_exports():
    import torchdyno.training as training

    for name in ["Learner", "FitResult", "BackpropTrainer"]:
        assert hasattr(training, name), name
