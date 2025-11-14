.PHONY: test
test:
	export TESTCONTAINERS_SESSION_ID=$$(uuidgen)
	docker build -t pyworker-test:latest -f docker/Dockerfile.worker.test .
	pytest -q

