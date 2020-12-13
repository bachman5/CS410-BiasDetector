WORKING_DIR = $(shell cd ../ && pwd)

PYTHON      = ${WORKING_DIR}/SVM_Model/.venv/bin/python3
PIP         = ${WORKING_DIR}/SVM_Model/.venv/bin/pip3
PACKAGES    = ${WORKING_DIR}/SVM_Model/requirements.txt

CORPUS_ALL_FF = ${WORKING_DIR}/SVM_Model/work2vec/data/corpus-all.txt


install:
	python -m venv .venv
	chmod +x .venv/bin/activate
	. .venv/bin/activate
	#${PYTHON} --version
	#${PIP} --version
	${PIP} install -r ${PACKAGES}
	cd ${WORKING_DIR}/SVM_Model/glove_genvecs && make && cd ${WORKING_DIR}

data:
	cd ${WORKING_DIR}; \
	. ./SVM_Model/.venv/bin/activate; \
	${PYTHON} -m SVM_Model.work2vec.prepare_data; \
	cp ${CORPUS_ALL_FF} ${WORKING_DIR}/SVM_Model/glove_genvecs/text8; \
	cd ${WORKING_DIR}/SVM_Model/glove_genvecs && ./demo.sh && cd ${WORKING_DIR}; \
	cp -f ${WORKING_DIR}/SVM_Model/glove_genvecs/vectors.magnitude ${WORKING_DIR}/SVM_Model/work2vec/corpus/; \

train:
	cd ${WORKING_DIR}; \
	${PYTHON} -m SVM_Model.work2vec.main --train

test:
	cd ${WORKING_DIR}; \
	. SVM_Model/.venv/bin/activate; \
	${PYTHON} -m SVM_Model.work2vec.main --test


