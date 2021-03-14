mkdir lib

git clone https://github.com/google/benchmark.git lib/benchmark
git clone https://github.com/google/googletest.git lib/benchmark/googletest

cd lib/benchmark

cmake -E make_directory "build"
cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release ../
cmake --build "build" --config Release
