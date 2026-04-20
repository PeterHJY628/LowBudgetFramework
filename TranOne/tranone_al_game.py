"""
ALGame subclass for TranOne experiments: allow zero initial labeled samples.

The upstream ALGame.reset() always calls _fit_classifier(); with an empty labeled
pool that breaks. Here we skip training until at least one sample is labeled.
"""
from core.environment import ALGame


class TranOneALGame(ALGame):
    """Same as ALGame, but skip the first classifier fit when the labeled pool is empty."""

    def reset(self, *args, **kwargs):
        with torch.no_grad():
            self.n_interactions = 0
            self.added_images = 0
            if not self._features_cached:
                self.classifier = self.dataset.get_classifier(self.model_rng)
                self.classifier = self.classifier.to(self.device)
                self._cache_features()
                self.initial_weights = self.classifier.state_dict()
            else:
                self.classifier.load_state_dict(self.initial_weights)
            self.optimizer = self.dataset.get_optimizer(self.classifier)
            self.reset_al_pool()
        if self.x_labeled.shape[0] == 0:
            self.current_test_accuracy = 0.0
            self.current_test_loss = float("nan")
        else:
            self._fit_classifier(from_scratch=True)
        self.initial_test_accuracy = self.current_test_accuracy
        return self.create_state()
