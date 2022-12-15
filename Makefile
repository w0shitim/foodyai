# ----------------------------------
#          PACKAGE ACTION
# ----------------------------------

get_model:
		python -c 'from foodyai.gc_bucket.load_model import get_model; get_model()'

get_config:
		python -c 'from foodyai.gc_bucket.load_model import get_config; get_config()'

get_class:
		python -c 'from foodyai.gc_bucket.data import get_class; get_class()'

get_annot:
		python -c 'from foodyai.gc_bucket.data import get_annotations; get_annotations()'

run_predict:
		python -c 'from foodyai.interface.main import predict; predict("./raw_data/009624.jpg")'

run_api:
	uvicorn foodyai.api.fast:app --reload


# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* foodyai/*.py

black:
	@black scripts/* foodyai/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr foodyai-*.dist-info
	@rm -fr foodyai.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)
