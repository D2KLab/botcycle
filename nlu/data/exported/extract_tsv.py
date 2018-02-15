import json
import csv
import itertools
import traceback
import dateutil.parser
import datetime
import plac

def main(lang):
    # for old dumps use '/messages_heroku.json' and '/messages_heroku_slack.json'
    with open(lang + '/messages_heroku_slack.json') as json_file:
        messages_raw = json.load(json_file)
    # for old dumps use '/nlu_history_heroku.json' and '/nlu_history_heroku_slack.json'
    with open(lang + '/nlu_history.json') as json_file:
        nlu_raw = json.load(json_file)
    try:
        with open(lang + '/stats.json') as json_file:
            stats = json.load(json_file)
    except:
        stats = {}
    # for old dumps, earlier in time, use None directly
    last_update = stats.get('last_update', None)
    if last_update:
        last_update = dateutil.parser.parse(last_update)
    newest_update = last_update
    
    nlu_lookup = {}

    entity_names = set()

    for nlu_entry in nlu_raw:
        intent = nlu_entry.get('intent', None)
        if intent:
            intent = intent['value']
        entities = nlu_entry.get('entities', None)
        try:
            text = nlu_entry.get('text', None) or nlu_entry['_text']
        except:
            pass
        nlu_lookup[text] = {'intent': intent, 'entities': entities}
        if entities:
            entity_names.update([k for k,v in entities.items()])

    # perform a group by: 1 get contiguous by chat_id, 2 group by
    messages_raw.sort(key=lambda m: m['chat_id'])
    sessions = []
    for key, values in itertools.groupby(messages_raw, lambda m: m['chat_id']):
        messages = []
        for m in values:
            try:
                date = dateutil.parser.parse(m.get('time',{}).get('$date', None))
            except:
                date = datetime.datetime.fromtimestamp(m.get('time', {}).get('$date', 0)/1000)
            if not last_update or date > last_update:
                messages.append(m)
                if not newest_update or date > newest_update:
                    newest_update = date
        if (messages):
            print('chat id', key, 'has', len(messages), 'new messages')
            sessions.append(messages)
    
    print('latest message at', newest_update)

    entity_names.remove('intent')
    entity_names = sorted(list(entity_names))
    
    field_names = ['role', 'text', 'intent'] + entity_names

    with open(lang + '/tabular.tsv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter='\t')

        if not last_update:
            # write the header only once
            writer.writeheader()
        for s in sessions:
            for m in s:
                try:
                    role = 'u' if m.get('type', 'request') == 'request' else 'b'
                    text = m['text']
                except:
                    print(m)
                    traceback.print_exc()
                row = {
                    'role': role,
                    'text': text
                }
                nlu_out = nlu_lookup.get(text, None)
                if nlu_out:
                    row['intent'] = nlu_out['intent']
                    if nlu_out['entities']:
                        #if not isinstance(nlu_out['entities'], dict):
                        #    nlu_out['entities'] = nlu_out['entities'][0]
                        for k,v in nlu_out['entities'].items():
                            try:
                                if not isinstance(v, dict):
                                    v = v[0]
                                row[k] = v['value']
                            except:
                                print(k,v)
                                traceback.print_exc()
                writer.writerow(row)

            writer.writerow({})

    # update statistics
    stats['last_update'] = newest_update.isoformat()
    with open(lang + '/stats.json', 'w') as json_file:
        json.dump(stats, json_file)



if __name__ == '__main__':
    plac.call(main)