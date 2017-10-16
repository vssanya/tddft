all: build_core build_wrapper

build_wrapper:
	python setup.py build_ext -b tdse

build_core:
	make -C build
