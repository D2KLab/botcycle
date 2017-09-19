import requests

graph_endpoint = 'https://graph.facebook.com/v2.10'


def __getHeader(token):
    return {'Authorization': 'Bearer {}'.format(token)}


def __get_unpaged(endpoint, params, token):
    results = []
    _params = params.copy()
    params['limit'] = 100
    # Facebook API uses pagination for likes, but we want them all
    partial = requests.get(endpoint, params=_params,
                           headers=__getHeader(token)).json()
    
    while partial['paging'].get('next', None) != None:
        results.extend(partial['data'])
        partial = requests.get(partial['paging']['next'],
                               headers=__getHeader(token)).json()

    return results


def get_user_likes(token):
    params = {'fields': 'location,category_list,name'}
    results = __get_unpaged(graph_endpoint + '/me/likes', params, token)

    return results


def get_user_profile(token):
    params = {'fields': 'age_range,gender,location'}
    return requests.get(graph_endpoint + '/me', params=params, headers=__getHeader(token)).json()


def get_user_tagged_places(token):
    params = {'fields': 'place{category_list,name,location}'}
    results = __get_unpaged(graph_endpoint + '/me/tagged_places', params, token)
    return results
