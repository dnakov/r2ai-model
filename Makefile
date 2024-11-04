TSVFILE=data/radare2/pending/claude-numbers2.tsv

all:
	./review-pending.sh "${TSVFILE}"
