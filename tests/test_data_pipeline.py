from data.datasets import stratified_split_indices


def test_cifar_stratified_split_is_deterministic_and_complete():
    labels = [label for label in range(100) for _ in range(500)]
    train_a, validation_a = stratified_split_indices(labels, seed=42)
    train_b, validation_b = stratified_split_indices(labels, seed=42)
    assert train_a == train_b
    assert validation_a == validation_b
    assert len(train_a) == 45000
    assert len(validation_a) == 5000
    assert set(train_a).isdisjoint(validation_a)
    assert set(train_a) | set(validation_a) == set(range(50000))