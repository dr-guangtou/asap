from asap import fitting

config_file = 'asap_test_4.yaml'

results = fitting.fit_asap_model(config_file, use_global=True, verbose=True)
