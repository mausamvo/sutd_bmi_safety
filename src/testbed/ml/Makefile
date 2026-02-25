check:
	uv run python tools/check_data.py ../sutd_bmi_safety_data/4sensors/data_1107/5g/

	uv run python tools/check_data.py ../sutd_bmi_safety_data/4sensors/data_1107/unseen/

	uv run python tools/check_data.py ../sutd_bmi_safety_data/4sensors/data_1107/microwave/

	uv run python tools/check_data.py ../sutd_bmi_safety_data/4sensors/data_1107/low_frequency/

	uv run python tools/check_data.py ../sutd_bmi_safety_data/4sensors/data_1107/baseline/
	
	uv run python tools/check_data.py ../sutd_bmi_safety_data/4sensors/data_1107/laser/

DATA ?= ../sutd_bmi_safety_data/4sensors/data_1107/combined.csv

train:
	uv run python semg_train_mlp.py --data $(DATA)
	
run_test:
	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/baseline/*.csv" >> results.md

	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/unseen/*.csv" >> results.md

	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/laser/*.csv" >> results.md

	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/microwave/*.csv" >> results.md

	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/5g/*.csv" >> results.md

	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/low_frequency/*.csv" >> results.md

	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/bass25hz/*.csv" >> results.md

	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/bass20hz/*.csv" >> results.md