import pybikes
import schedule

bike_info = {}
to_update = []

def search_nearest(position, search_type):

    info = get_city_cached(position)

    if not info:
        return None, None

    if search_type == 'bikes':
        results_set = info['with_bikes']

    else:
        results_set = info['with_slots']

    distance_sq = float('inf')
    best = -1
    #print("results_set has size: " + str(len(results_set)))
    for idx, val in enumerate(results_set):
        d2 = (position['latitude']-val.latitude) **2 + (position['longitude']-val.longitude) **2
        if d2 < distance_sq:
            distance_sq = d2
            best = idx

    if best is -1:
        return info['city'], None

    return info['city'], results_set[best]

# working on global variables?? SRSLY?
def update_data(which_to_update):
    print('update_data called on : ' + str(which_to_update))
    result = {}
    for value in which_to_update:
        print('going to get info on ' + value)
        try:
            bikeshare = pybikes.get(value)
            bikeshare.update()

            result[value] = {
                'city': bikeshare.meta['city'],
                'stations': {x.name:x for x in bikeshare.stations},
                'with_bikes': [station for station in bikeshare.stations if station.bikes>0],
                'with_slots': [station for station in bikeshare.stations if station.free>0]
            }

        except Exception as e:
            print('something bad while getting info for ' + value + ': ' + str(e) + '\n, discarding this city')
            which_to_update.remove(value)



    return result

    """
    torino_bikeshare = pybikes.get('to-bike')
    torino_bikeshare.update()
    torino_stations = {x.name:x for x in torino_bikeshare.stations}
    stations_with_bikes = [station for station in torino_bikeshare.stations if station.bikes>0]
    stations_with_free = [station for station in torino_bikeshare.stations if station.free>0]"""

def get_city_cached(position):
    global bike_info, to_update

    tag = nearest_city_find(position)
    result = bike_info.get(tag, None)
    if not result:
        to_update.append(tag)
        bike_info = update_data(to_update)
        # now the city must be there or something bad happened
        result = bike_info.get(tag, None)

    return result

def nearest_city_find(position):
    where_to_search = {}
    for schema in pybikes.get_all_data():
        data = pybikes.get_data(schema)
        instances = data.get('instances', None)
        if not instances:
            instances = []
            for key, value in data['class'].items():
                instances.extend(value['instances'])

        for instance in instances:
            where_to_search[instance['tag']] = instance['meta']

    # now where_to_search contains a map (tag, {latitude, longitude, ...})
    #pprint(where_to_search)

    distance_sq = float('inf')
    best = -1
    #print("results_set has size: " + str(len(results_set)))
    for key, value in where_to_search.items():
        d2 = (position['latitude']-value['latitude']) **2 + (position['longitude']-value['longitude']) **2
        if d2 < distance_sq:
            distance_sq = d2
            best = key

    #print("nearest is " + best)
    return best, where_to_search[best]


def update():
    global bike_info
    try:
        bike_info = update_data(to_update)

    except Exception as e:
        print('something bad happened: ' + str(e))

# schedule execution of update every minute
schedule.every(1).minutes.do(update)