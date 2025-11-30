PY=python

WINDOWS=2 3 4 5
FOLDS=0 1 2 3 4

.PHONY: data normalize train_linear train_extra aggregate report full

data:
	$(PY) scripts/process_data_full.py

normalize: data
	$(PY) scripts/build_pipeline.py

train_linear:
	@for w in $(WINDOWS); do \
	  for f in $(FOLDS); do \
	    $(PY) scripts/train_linear.py --window $$w --fold $$f; \
	  done; \
	done

train_extra:
	$(PY) scripts/train_tree_nn.py --window 3 --fold 0 --model rf --params '{"n_estimators":300,"max_depth":12}'
	$(PY) scripts/train_tree_nn.py --window 3 --fold 0 --model mlp --params '{"hidden_layer_sizes":[128,64],"max_iter":400}'

aggregate:
	$(PY) scripts/aggregate_metrics.py

report:
	$(PY) scripts/report_plots.py

full: normalize train_linear train_extra aggregate report

