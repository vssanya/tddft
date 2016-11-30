all: build_wrapper


build_wrapper:
	python3.5 setup.py build_ext -b tdse
