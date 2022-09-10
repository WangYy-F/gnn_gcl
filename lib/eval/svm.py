from sklearn.svm import LinearSVC, SVC
from lib.eval import BaseSKLearnEvaluator


class SVMEvaluator(BaseSKLearnEvaluator):
    def __init__(self, linear=True, params=None):
        if linear:
            self.evaluator = LinearSVC(dual=False,max_iter=10000)
        else:
            self.evaluator = SVC(max_iter=100000)
        if params is None:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        super(SVMEvaluator, self).__init__(self.evaluator, params)
