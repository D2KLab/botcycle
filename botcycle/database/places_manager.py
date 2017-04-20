import requests
from pprint import pprint

class Places_manager:

    def __init__(self, tokens, db_conn):
        self.key = tokens['google_places']
        self.db_conn = db_conn

    def getPlaceByName(self, place_name):
        result = {}
        # TODO the location is the one of Torino, be more general
        params = {'key': self.key, 'rankby': 'distance', 'location': '45.066667,7.7', 'keyword': place_name}
        response = requests.get("https://maps.googleapis.com/maps/api/place/nearbysearch/json", params = params).json()
        result['latitude'] = response['results'][0]['geometry']['location']['lat']
        result['longitude'] = response['results'][0]['geometry']['location']['lng']
        result['name'] = response['results'][0]['name']
        result['placeId'] = response['results'][0]['id']
        result['types'] = response['results'][0]['types']

        self.savePlaceIfAbsent(result)

        return result

    def getPlaceByLocation(self, location):
        # TODO should get only relevant places
        result = {}
        params = {'key': self.key, 'rankby': 'distance', 'location': "{0},{1}".format(location['latitude'], location['longitude'])}
        response = requests.get("https://maps.googleapis.com/maps/api/place/nearbysearch/json", params = params).json()

        result['latitude'] = response['results'][0]['geometry']['location']['lat']
        result['longitude'] = response['results'][0]['geometry']['location']['lng']
        result['name'] = response['results'][0]['name']
        result['placeId'] = response['results'][0]['id']
        result['types'] = response['results'][0]['types']

        self.savePlaceIfAbsent(result)

        return result

    def savePlaceIfAbsent(self, place):
        place_arr = (place['placeId'], place['name'], place['latitude'], place['longitude'])
        self.db_conn.execute("INSERT OR IGNORE INTO Place (id, name, description, lat, lng, category) VALUES (?, ?, NULL, ?, ?, NULL)", place_arr)
        self.db_conn.commit()

    def saveEvent(self, event):
        pass
