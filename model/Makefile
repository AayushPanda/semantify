DOC_DIR=data/docs
OUTPUT_DIR=outputs

.PHONY: all floppy_img kernel bootloader clean ${BUILD_DIR}

get_docs_random: ${DOC_DIR} venv
	python scrape_docs.py

${DOC_DIR}:
	mkdir -p ${DOC_DIR}

${OUTPUT_DIR}:
	mkdir -p ${OUTPUT_DIR}

clean-docs:
	rm -rf ${DOC_DIR}/*

clean-output:
	rm -rf ${OUTPUT_DIR}/*

venv:
	python -m venv .venv
	.venv/bin/activate
	pip install -r requirements.txt