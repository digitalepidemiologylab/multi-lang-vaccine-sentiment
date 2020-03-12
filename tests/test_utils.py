import sys; sys.path.append('..');
import uuid

from vac_utils import get_predictions_output

def test_predictions_output():
    experiment_id = str(uuid.uuid4())
    probabilities = [[0.1, 0.4, 0.5], [0.1, 0.4, 0.5], [0.1, 0.4, 0.5]]
    y_true = [0, 1, 0]
    label_list = ['positive', 'neutral', 'negtive']
    guids = [str(i) for i in range(3)]
    label_mapping = dict(zip(range(len(label_list)), label_list))
    output = get_predictions_output(experiment_id, guids, probabilities, y_true, label_mapping)
    assert len(output) == 2
    assert list(output['prediction_output'].keys()) == guids
    assert output['Experiment_Id'] == experiment_id


if __name__ == "__main__":
    import pytest
    pytest.main()
