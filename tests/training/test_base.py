from torchdyno.training.base import FitResult, Learner


def test_fitresult_defaults_are_empty_dicts():
    r = FitResult()
    assert r.history == {} and r.best == {} and r.extras == {}


def test_fitresult_holds_values():
    r = FitResult(history={"loss": [1.0, 0.5]}, best={"l2": 1e-6}, extras={"A": 1})
    assert r.history["loss"] == [1.0, 0.5]
    assert r.best["l2"] == 1e-6
    assert r.extras["A"] == 1


def test_learner_is_structural():
    class Ok:
        def fit(self, model, train, val=None):
            return FitResult()

    class NotOk:
        pass

    assert isinstance(Ok(), Learner)
    assert not isinstance(NotOk(), Learner)
