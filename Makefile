.PHONY: install test figures clean lint format

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

figures:
	MPLBACKEND=Agg python paper/figures/scripts/fig_restitution_map.py --no-show
	MPLBACKEND=Agg python paper/figures/scripts/fig_profiles_approach_spread.py --no-show
	MPLBACKEND=Agg python paper/figures/scripts/fig_energy_budget.py --no-show
	MPLBACKEND=Agg python paper/figures/scripts/fig_time_series.py --no-show
	MPLBACKEND=Agg python paper/figures/scripts/fig_central_gap_scaling.py --no-show
	MPLBACKEND=Agg python paper/figures/scripts/fig_exponent_summary.py --no-show
	MPLBACKEND=Agg python paper/figures/scripts/fig_gap_velocity_field.py --no-show

sample:
	python data/extract_sample.py

lint:
	ruff check src tests scripts data

format:
	ruff format src tests scripts data

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	rm -rf paper/figures/output/*.pdf
