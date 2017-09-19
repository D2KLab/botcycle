from . import fb_graph
from . import persistence

def add_data_from_login(chat_id, token):
    """This gets all the useful data about the user and saves the information in the db"""
    persistence.save_user_token(chat_id, token)
    # TODO consider doing that in separate thread
    update_facebook_data(token)

def update_facebook_data(token):
    """This updates the profile of a single user"""
    # TODO consider facebook id as unique identifier (call fb_graph.get_me)
    profile = fb_graph.get_user_profile(token)
    likes = fb_graph.get_user_likes(token)
    tagged_places = fb_graph.get_user_tagged_places(token)

    print('done, collected profile, {} likes, {} tagged_places'.format(len(likes), len(tagged_places)))
    # TODO save somewhere