import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class HonestButCuriousServer:
    """
    Simulates an adversary logging client weight updates to perform 
    Subject Re-identification attacks.
    """
    def __init__(self):
        self.snooped_features = []
        self.snooped_labels = []
        # A Random Forest is excellent at finding non-linear patterns in 1D arrays
        self.attacker_model = RandomForestClassifier(n_estimators=100, random_state=42)

    def extract_fingerprint(self, weight_diff):
        """Flattens the weight dictionary into a single 1D numpy array."""
        features = []
        for name, tensor in weight_diff.items():
            if tensor.dtype in [torch.float32, torch.float64]:
                features.append(tensor.cpu().numpy().flatten())
        return np.concatenate(features)

    def snoop_update(self, weight_diff, client_id):
        """Logs the incoming weight update before aggregation."""
        fingerprint = self.extract_fingerprint(weight_diff)
        self.snooped_features.append(fingerprint)
        self.snooped_labels.append(client_id)

    def perform_reidentification_attack(self):
        """Trains the attack model and returns the attack success rate (accuracy)."""
        if len(self.snooped_features) < 10:
            return 0.0  # Not enough data intercepted yet

        X = np.array(self.snooped_features)
        y = np.array(self.snooped_labels)

        # Split the snooped data to test if the attacker can predict unseen updates
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.attacker_model.fit(X_train, y_train)
        predictions = self.attacker_model.predict(X_test)
        
        attack_accuracy = accuracy_score(y_test, predictions)
        return attack_accuracy * 100.0