import model_pipeline


def test_functions_exist():
    assert callable(model_pipeline.prepare_data)
    assert callable(model_pipeline.train_model)
    assert callable(model_pipeline.evaluate_model)
    assert callable(model_pipeline.save_model)
    assert callable(model_pipeline.load_model)
