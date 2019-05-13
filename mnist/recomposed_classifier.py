from common import *


# def activation_score(z_mean, z_log_var, pattern):
#     z_var = np.diag(np.exp(z_log_var)[0])
#     p_mean, p_var = pattern
#     return Bhattacharyya_coeff(np.ravel(z_mean), z_var, p_mean, p_var)


class RecomposedClassifier(object):
    def __init__(self, specilized_clfs, num_labels):
        self.specialized_clfs = specilized_clfs
        self.num_labels = num_labels

    def p_label(self, encoder, x):
        z_mean, z_log_var, _ = encoder.predict(x)
        sum_scores = 0

        p_label = np.array([0.] * self.num_labels)
        for clf in self.specialized_clfs:
            activation = activation_score(z_mean, z_log_var, clf.pattern)
            p_label += clf.p_label * activation
            sum_scores += activation

        return p_label / sum_scores