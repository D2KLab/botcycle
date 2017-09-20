from . import fb_graph
from . import persistence
from . import foursquare


def add_data_from_login(chat_id, facebook_id, access_token):
    """This gets all the useful data about the user and saves the information in the db"""
    persistence.save_facebook_token(chat_id, facebook_id, access_token)
    # TODO schedule this operation in separate thread
    # update_facebook_profile(facebook_id)


def update_facebook_profile(facebook_id):
    # TODO add check to avoid update of new data
    facebook_profile = persistence.get_facebook_user(facebook_id)
    likes = fb_graph.get_user_likes(facebook_profile['access_token'])
    tagged_places = fb_graph.get_user_tagged_places(
        facebook_profile['access_token'])

    # the list of ids of the liked pages
    likes_id = map(lambda el: el['id'], likes)

    print('done, collected profile, {} likes, {} tagged_places'.format(
        len(likes), len(tagged_places)))

    foursquare_liked_places = __find_places_matches(likes)
    print('likes matches: {}'.format(len(foursquare_liked_places)))
    foursquare_tagged_places = __find_places_matches(tagged_places)
    print('tagged places matches: {}'.format(len(foursquare_tagged_places)))

    # TODO get the categories


def __find_places_matches(facebook_pages):
    """search march of facebook pages to obtain a list of foursquare venues"""
    results = []
    for page in facebook_pages:
        location = page.get('location', {})
        lat, lng = location.get(
            'latitude', None), location.get('longitude', None)
        if lat and lng:
            # a place may be attached
            place = foursquare.match(page['name'], lat, lng)
            if place:
                results.append(place)

    return results