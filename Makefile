TSVFILE=data/radare2/pending/claude-numbers2.tsv
# TSVFILE=data/radare2/pending/r2gpt-advent.tsv

all:
	./review-pending.sh "${TSVFILE}"
