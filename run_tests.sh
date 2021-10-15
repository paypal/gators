# pytest -v --doctest-modules --cov-report html:cov_html --cov=gators gators
pytest -v --cov-report html:cov_html --cov=gators gators
open cov_html/index.html
