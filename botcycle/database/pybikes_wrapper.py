import pybikes
import time

class Pybikes_wrapper:
    def __init__(self, city_name):
        self.city_name = city_name
        self.pybike_data = pybikes.get('to-bike')
        self.update()
        #torino_stations = {x.name:x for x in torino_bikeshare.stations}
        #stations_with_bikes = [station for station in torino_bikeshare.stations if station.bikes>0]
        #stations_with_free = [station for station in torino_bikeshare.stations if station.free>0]

    # get all the stations
    def get_stations(self):
        return self.stations_dict

    # get a specific station given the name
    def get_station(self, station_name):
        return self.stations.get(station_name, None)

    # feature=None -> search nearest station
    # feature='bike' -> search nearest station with at least threshold bikes available
    # feature='slot' -> search nearest station with at least threshold slots available
    def search_nearest(self, position, feature=None, treshold=1):
        print('search_nearest position=' + str(position))
        where_to_search = self.pybike_data.stations
        if feature == 'bike':
            where_to_search = self.stations_with_bikes

        elif feature == 'slot':
            where_to_search = self.stations_with_slots

        distance_sq = float('inf')
        best = -1
        for idx, val in enumerate(where_to_search):
            d2 = (position['latitude']-val.latitude) **2 + (position['longitude']-val.longitude) **2
            if d2 < distance_sq:
                distance_sq = d2
                best = idx

        return where_to_search[best]

    def update(self):
        self.pybike_data.update()
        self.updated = time.time()
        self.stations_dict = {x.name:x for x in self.pybike_data.stations}
        self.stations_with_bikes = [station for station in self.pybike_data.stations if station.bikes>0]
        self.stations_with_slots = [station for station in self.pybike_data.stations if station.free>0]
