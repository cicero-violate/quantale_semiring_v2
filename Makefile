.PHONY: run check clean cuda-smoke

run:
	cargo run --release

check:
	cargo check

cuda-smoke:
	nvcc -O3 --std=c++17 cuda/quantale_world.cu -o /tmp/quantale_world_smoke -lcudart

clean:
	cargo clean
