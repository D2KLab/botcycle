import json


def load_expressions():
    """Returns the expressions_file loaded from JSON"""
    with open('../data/BotCycle/expressions.json') as expressions_file:
        return json.load(expressions_file)


def load_intents():
    """Returns a list of names of the intents"""
    with open('../data/BotCycle/entities/intent.json') as intents_file:
        intents = json.load(intents_file)
        return list(map(lambda x: x['value'], intents['data']['values']))


def get_train_data(expressions):
    """Returns a list of tuples (text, intent)"""
    array = expressions['data']

    result = []
    # substitute the entities with their name
    for sentence in array:
        text = sentence['text']
        intent = None
        for entity in sentence['entities']:
            if entity['entity'] != 'intent':
                text = text.replace(entity['value'].strip(
                    '"'), entity['entity'].upper())
            else:
                intent = entity['value'].strip('"')

        result.append((text, intent))

    # print(result)
    return result


def main():
    expressions = load_expressions()
    return get_train_data(expressions)


if __name__ == '__main__':
    print(main())
