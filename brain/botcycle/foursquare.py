import os
import requests

FOURSQUARE_ENDPOINT = 'https://api.foursquare.com/v2/venues'
common_params = {
    'v': '20170920',  # version
    'client_id': os.environ['FOURSQUARE_CLIENT_ID'],
    'client_secret': os.environ['FOURSQUARE_CLIENT_SECRET']
}


def match(name, lat, lng):
    """search a place with the provided name and position, result can be None if no match is found"""
    extra_params = {
        'intent': 'match',
        'll': '{},{}'.format(lat, lng),
        'query': name
    }
    params = {**common_params, **extra_params}
    response = requests.get(FOURSQUARE_ENDPOINT +
                            '/search', params=params).json()
    venues = response['response']['venues']
    if venues:
        return venues[0]
    else:
        return None


def get_top_picks(lat, lng):
    """get top picks near provided position"""
    extra_params = {
        'll': '{},{}'.format(lat, lng),
        'section': 'topPicks'
    }
    params = {**common_params, **extra_params}
    response = requests.get(FOURSQUARE_ENDPOINT +
                            '/explore', params=params).json()

    venues = response['response']['groups'][0]['items']
    venues = list(map(lambda venue: venue['venue'], venues))
    for venue in venues:
        venue['url'] = 'https://foursquare.com/v/{}?ref={}'.format(venue['id'], common_params['client_id'])

    return venues
