mongoexport --host mongodb --db botcycle --collection messages --jsonArray --out en/messages.json
mongoexport --host mongodb --db botcycle --collection nlu_history --jsonArray --out en/nlu_history.json
mongoexport --host mongodb --db botcycle_it --collection messages --jsonArray --out it/messages.json
mongoexport --host mongodb --db botcycle_it --collection nlu_history --jsonArray --out it/nlu_history.json

# old dumps
#mongoexport --db botcycle_heroku --collection messages --jsonArray --out en/messages_heroku.json
#mongoexport --db botcycle_heroku --collection nlu_history --jsonArray --out en/nlu_history_heroku.json
#mongoexport --db botcycle_heroku_slack --collection messages --jsonArray --out en/messages_heroku_slack.json
#mongoexport --db botcycle_heroku_slack --collection nlu_history --jsonArray --out en/nlu_history_heroku_slack.json

python extract_tsv.py en
#python extract_tsv.py it