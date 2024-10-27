# r2ai-model

Collection of data sources to generate a dataset for training and finetuning LLM models to use radare2.

## Organization

Dataset is stored in Q/A form (Question/Answer) separating them by tabs (TSV) where the question is phrased in English and the answer is an r2 oneliner to be executed by r2ai in auto mode.

* / -> root directory, scripts to generate raw QA
* `data/radare2_ok.tsv` -> validated statements
* `data/radare2_todo.tsv` -> unanswered questions
* data/Attic/ -> already processed files
* data/sources -> unfiltered data sources to be used to generate questions
