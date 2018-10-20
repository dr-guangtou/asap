from asap import fitting

config_file = 'asap_test_1.yaml'

results = fitting.fit_asap_model(config_file, use_global=True, verbose=True)
