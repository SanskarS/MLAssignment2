from models.logistic_regression import run as run_logistic_regression
from models.decision_tree import run as run_decision_tree
from models.knn import run as run_knn
from models.naive_bayes import run as run_naive_bayes
from models.random_forest import run as run_random_forest

RUNNERS = {
    'logistic_regression': run_logistic_regression,
    'decision_tree': run_decision_tree,
    'knn': run_knn,
    'naive_bayes': run_naive_bayes,
    'random_forest': run_random_forest,
}
