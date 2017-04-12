"""
This class is responsible for resolving the entities. There are two kinds of resolution:
- from named place (e.g. home/work) to place (table Place)
- from place name to position --> using places API (and checking if need insertion in table Place or not)

Two kinds of place identification are possible:
- from name to placeId: location=<city-position>&rankby=distance&keyword=<name>
- from position to placeId: sort by proximity location=<location>&rankby=distance or location=<location>&radius=500&type=point_of_interest ?

The structure of entity is:
{
    'confidence': 0.9760548082014214,
    'suggested': True,
    'type': 'location',
    'value': 'porta palazzo'
}

The resolution of places adds:
'resolved': {
    'location': {
        'latitude': XXX,
        'longitude': YYY
    },
    'place': {
        'placeID'
        // and so on for all the columns of the places table
    }
}
"""

class Resolver:
    def __init__(self):
        pass
        #self.arg = arg

    def resolve(self, chat_id, entities, utils):
        # for each entity -> from name to placeID
        for key, entity in entities.items():
            if entity['type'] == 'location':
                place = utils['data_manager'].places_manager.getPlaceByName(entity['value'])
                entity['resolved'] = {
                    'location': {'latitude': place['latitude'], 'longitude': place['longitude']},
                    'place': place
                }
        # also add the user position to the resolved_entities
        return entities

    def resolve_location(self, location, utils):
        # create an entity that contains location 'as-is' (will be used for precise search) + resolved placeId (will be used by personalization&co)
        place = utils['data_manager'].places_manager.getPlaceByLocation(location)
        result = {
            'type': 'location',
            'resolved': {
                'location': {'latitude': location['latitude'], 'longitude': location['longitude']},
                'place': place
            }
        }
        return result
