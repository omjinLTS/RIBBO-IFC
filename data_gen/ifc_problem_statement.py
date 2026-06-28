from vizier import pyvizier as vz
from problems.ifc import IFCTorch


def problem_statement(search_space_id, dataset_id):
    lb, ub = 0.0, 1.0
    f = IFCTorch(search_space_id, dataset_id, lb=lb, ub=ub)  # dim inferred from search_space_id

    problem = vz.ProblemStatement()
    root = problem.search_space.root
    for i in range(f.dim):
        root.add_float_param('x{}'.format(i), lb, ub)
    metric = vz.MetricInformation(
        name='obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE,
    )
    problem.metric_information.append(metric)
    return problem, f
