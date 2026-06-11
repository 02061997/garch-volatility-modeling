.PHONY: test reproduce-smoke reproduce-results
test:
	uv run pytest
reproduce-smoke:
	PYTHONPATH=src uv run python -m volatilitylab.run --smoke
reproduce-results:
	PYTHONPATH=src uv run python -m volatilitylab.run
