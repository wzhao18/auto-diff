.PHONY: lib, pybind, clean, format, all

all: lib


lib:
	@mkdir -p build
	@cd build; cmake ..
	@cd build; $(MAKE)

test:
	NEEDLE_BACKEND=np python3 -m pytest \
		tests/test_autograd_hw.py \
		tests/test_data.py
	NEEDLE_BACKEND=np python3 -m pytest \
		tests/test_nn_and_optim.py
	NEEDLE_BACKEND=nd python3 -m pytest \
		tests/test_cifar_ptb_data.py \
		tests/test_conv.py \
		tests/test_nd_backend.py \
		tests/test_ndarray.py \
		tests/test_sequence_models.py

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so
