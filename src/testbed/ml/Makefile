check:
	uv run python tools/check_data.py ../sutd_bmi_safety_data/4sensors/data_1107/5g/

	uv run python tools/check_data.py ../sutd_bmi_safety_data/4sensors/data_1107/unseen/

	uv run python tools/check_data.py ../sutd_bmi_safety_data/4sensors/data_1107/microwave/

	uv run python tools/check_data.py ../sutd_bmi_safety_data/4sensors/data_1107/low_frequency/

	uv run python tools/check_data.py ../sutd_bmi_safety_data/4sensors/data_1107/baseline/

	uv run python tools/check_data.py ../sutd_bmi_safety_data/4sensors/data_1107/laser/

TRAIN_DATA_DIR ?= ../sutd_bmi_safety_data/4sensors/data_1107/baseline
DATA ?= ../sutd_bmi_safety_data/4sensors/data_1107/combined.csv
# removed wl from default features
FEATURES ?= mav,rms,var,zc,ssc

process:
	uv run python tools/process_data.py $(TRAIN_DATA_DIR)

CALIB_DATA ?= ../sutd_bmi_safety_data/4sensors/data_1107/unseen/*.csv
N_CALIB ?= 2

train:
	uv run python semg_train_mlp.py --data $(DATA) --features $(FEATURES)

calibrate:
	uv run python semg_calibrate.py --csv "$(CALIB_DATA)" --n_calib $(N_CALIB) --eval

clean_calib:
	rm -f semg_mlp_calibrated.pth

SVM_KERNEL ?= linear
SVM_C ?= 1.0

train_svm:
	uv run python semg_train_svm.py --data $(DATA) --features $(FEATURES) --kernel $(SVM_KERNEL) --C $(SVM_C)

run_test_svm:
	uv run python semg_infer_svm.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/baseline/**/*.csv" >> results_svm.md

	uv run python semg_infer_svm.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/unseen/*.csv" >> results_svm.md

	uv run python semg_infer_svm.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/laser/*.csv" >> results_svm.md

	uv run python semg_infer_svm.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/microwave/*.csv" >> results_svm.md

	uv run python semg_infer_svm.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/5g/*.csv" >> results_svm.md

	uv run python semg_infer_svm.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/low_frequency/*.csv" >> results_svm.md

	uv run python semg_infer_svm.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/bass25hz/*.csv" >> results_svm.md

	uv run python semg_infer_svm.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/bass20hz/*.csv" >> results_svm.md

run_test:
	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/baseline/**/*.csv" >> results.md

	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/unseen/*.csv" >> results.md

	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/laser/*.csv" >> results.md

	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/microwave/*.csv" >> results.md

	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/5g/*.csv" >> results.md

	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/low_frequency/*.csv" >> results.md

	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/bass25hz/*.csv" >> results.md

	uv run python semg_infer_mlp.py --mode offline --csv "../sutd_bmi_safety_data/4sensors/data_1107/bass20hz/*.csv" >> results.md