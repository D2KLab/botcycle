mongoexport --host mongodb --db botcycle --collection messages --jsonArray --out en/messages.json
mongoexport --host mongodb --db botcycle --collection nlu_history --jsonArray --out en/nlu_history.json
mongoexport --host mongodb --db botcycle_it --collection messages --jsonArray --out it/messages.json
mongoexport --host mongodb --db botcycle_it --collection nlu_history --jsonArray --out it/nlu_history.json

python extract_tsv.py en
python extract_tsv.py it