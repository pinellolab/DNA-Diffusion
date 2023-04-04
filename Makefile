.PHONY: clean lint requirements test dist publish bump_major_version \
bump_minor_version bump_patch_version fetch_new_major_version \
fetch_new_minor_version fetch_new_patch_version fetch_current_version \
bump_build_version fetch_new_build_version bump_release_version \
fetch_new_release_version

#################################################################################
# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# Goals of the Makefile is:
# 1. Compiling and linking source code
# 2. Generating documentation
# 3. Running tests
# 4. Packaging the application for distribution
# 5. Cleaning up intermediate files and artifacts

# To use it type: make $command

# TODO: fix the build process and the commands that accompany the build process
#################################################################################

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROJECT_NAME = dna-diffusion
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies 
requirements: test_environment
	conda install environments/conda/environment.yml

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Perform style checks using pylint [TODO: change with other linter if needed]
style_check:
	PYTHONPATH=src pylint -j 4 -d duplicate-code src/
	PYTHONPATH=src pylint -j 4 -d duplicate-code tests/*.py

## Perform type checks using mypy [TODO: change with other type-checker if needed]
type_check:
	mypy --namespace-packages --ignore-missing-imports --disallow-untyped-defs src/

## Perform style checks using pylint and type checks using mypy
lint: style_check type_check

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Run tests
test: requirements
	PYTHONPATH=src/ py.test --cov=src/ --cov-report html:cov_html_doctests --doctest-modules -v tests/

## [TODO: add commands for packaging the project]

## Format code
format:
	$(PYTHON_INTERPRETER) -m black \
		--line-length 80 \
		--target-version py38 \
		src/ tests/

## Format code
format_check:
	$(PYTHON_INTERPRETER) -m black --check \
		--line-length 80 \
		--target-version py38 \
		src/ tests/

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

# Custom function for fetching new/current version for release
define fetch_version
	@bump2version --allow-dirty --dry-run --list $(2) | grep $(1)_version | sed -r s,"^.*=",,
endef
