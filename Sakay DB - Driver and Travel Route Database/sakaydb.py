import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import os


class SakayDBError(ValueError):
    pass


class SakayDB():

    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Initializing dataframes to None
        self.trips_df = None
        self.drivers_df = None
        self.locations_df = None

    def add_trip(self,
                 driver,
                 pickup_datetime,
                 dropoff_datetime,
                 passenger_count,
                 pickup_loc_name,
                 dropoff_loc_name,
                 trip_distance,
                 fare_amount
                 ):
        """
        A method used for adding a single trip in the database. It appends
        the input trip data to the end of trips.csv, if it exists, or creates
        it, otherwise.
        The method returns the trip_id or raise a SakayDBError exception
        if the trip is already in trips.csv

        Parameters:
        ----------
        driver : str
            The trip driver as a string in Last name, Given name format
        pickup_datetime : str
            The pickup datetime as string with format "hh:mm:ss,DD-MM-YYYY"
        dropoff_datetime : str
            The dropoff datetime as string with format "hh:mm:ss,DD-MM-YYYY"
        passenger_count : int
            The number of passengers as integer
        pickup_loc_name : str
            The pickup location name as a string, (e.g., Pine View, Legazpi Village)
        dropoff_loc_name : str
            The dropoff location name as a string, (e.g., Pine View, Legazpi Village)
        trip_distance : float
            The distance in meters
        fare_amount : float
            The amount paid by passenger
        pickup_loc_id : int
            A number assigned to the pickup location. This is based on the
            locations.csv file
        dropoff_loc_id : int
            A number assigned to the drop-off location. This is based on the
            locations.csv file

        Returns:
        ----------
        trip_id: int
            A number assigned to the trip. Starting value is 1, next is
            2, 3 ... and so on. Only valid and non-existing trips will be
            returned with a new trip_id.
            Returns the trip_id or raise a SakayDBError exception if the trip
            is already in trips.csv
        """

        # initialize attributes
        self.driver = driver.strip()
        self.pickup_datetime = pickup_datetime
        self.dropoff_datetime = dropoff_datetime
        self.passenger_count = passenger_count
        self.trip_distance = trip_distance
        self.fare_amount = fare_amount

        # create the pickup_id and dropoff_id attributes
        df_locations = pd.read_csv(self.data_dir + '/locations.csv')

        filt_loc = (df_locations['loc_name'].str.lower()
                    == pickup_loc_name.lower().strip())
        self.pickup_id = int(df_locations[filt_loc]['location_id'].values[0])

        filt_loc = (df_locations['loc_name'].str.lower()
                    == dropoff_loc_name.lower().strip())
        self.dropoff_id = int(df_locations[filt_loc]['location_id'].values[0])

        # 1 trips_id
        try:
            df_trip = pd.read_csv(self.data_dir + '/trips.csv')
        except:
            trip_columns = np.array(['trip_id',
                                     'driver_id',
                                     'pickup_datetime',
                                     'dropoff_datetime',
                                     'passenger_count',
                                     'pickup_loc_id',
                                     'dropoff_loc_id',
                                     'trip_distance',
                                     'fare_amount'])

            trip_id = 1
            driver_id = 1

            first_entry = np.array([trip_id,
                                    driver_id,
                                    self.pickup_datetime,
                                    self.dropoff_datetime,
                                    self.passenger_count,
                                    self.pickup_id,
                                    self.dropoff_id,
                                    self.trip_distance,
                                    self.fare_amount,
                                    ])

            first_entry = first_entry.reshape(1, 9)
            df_trip = pd.DataFrame(data=first_entry, columns=trip_columns)
            df_trip.to_csv(self.data_dir + '/trips.csv', index=False)

        else:
            trip_id = len(df_trip.index) + 1
            new_entry = [trip_id,
                         self.driver,
                         self.pickup_datetime,
                         self.dropoff_datetime,
                         self.passenger_count,
                         self.pickup_id,
                         self.dropoff_id,
                         self.trip_distance,
                         self.fare_amount,
                         ]

            df_driver = pd.read_csv(self.data_dir + '/drivers.csv')
            last_name, given_name = self.driver.lower().split(', ')
            filt = ((df_driver['given_name'].str.lower() == given_name)
                    & (df_driver['last_name'].str.lower() == last_name))

            if df_driver[filt].empty:
                new_entry[1] = len(df_driver.index) + 1

            else:
                new_entry[1] = int(df_driver[filt]['driver_id'])

            df_trip_copy = df_trip.copy()
            new_entry_copy = new_entry.copy()

            if (df_trip_copy.iloc[:, 1:] == new_entry_copy[1:]).all(axis=1).any():
                raise SakayDBError
                print('Trip data already existing!')
                return

            else:
                df_trip.loc[len(df_trip.index)] = new_entry
                df_trip.to_csv(self.data_dir + '/trips.csv', index=False)

        # 2 driver_id
        try:
            df_driver = pd.read_csv(self.data_dir + '/drivers.csv')

        except:
            driver_columns = np.array(['driver_id',
                                       'given_name',
                                       'last_name'])
            driver_id = 1
            first_entry = np.array([driver_id,
                                    self.driver.split(', ')[1].title(),
                                    self.driver.split(', ')[0].title()]
                                   ).reshape(1, 3)

            df_driver = pd.DataFrame(data=first_entry, columns=driver_columns)
            df_driver.to_csv(self.data_dir + '/drivers.csv', index=False)

        else:
            driver_id = len(df_driver.index) + 1
            df_fullname = (df_driver['last_name'] + ', '
                           + df_driver['given_name'])
            if self.driver.lower() not in [x.lower() for x in
                                           df_fullname.values.tolist()]:
                new_entry = [driver_id,
                             self.driver.split(', ')[1].title(),
                             self.driver.split(', ')[0].title()
                             ]

                df_driver.loc[len(df_driver.index)] = new_entry
                df_driver.to_csv(self.data_dir + '/drivers.csv', index=False)

        return trip_id

    def add_trips(self, list_of_trips):
        """
        A method used for adding a multiple trips in the database. It accepts 
        a list of trips in the form of dictionaries. It appends each trip
        data to the end of trips.csv, if it exists, or creates it, otherwise.
        The method returns a list of the trip_ids of successfully added trips.
        If a trip is already in the database or has invalid or incomplete 
        information, it skips it and prints a warning. The trip index is the 
        zero-based index of the trip in the passed list of trips.

        Parameters:
        ----------
        driver : str
            The trip driver as a string in Last name, Given name format
        pickup_datetime : str
            The pickup datetime as string with format "hh:mm:ss,DD-MM-YYYY"
        dropoff_datetime : str
            The dropoff datetime as string with format "hh:mm:ss,DD-MM-YYYY"
        passenger_count : int
            The number of passengers as integer
        pickup_loc_name : str
            The pickup location name as a string, (e.g., Pine View, Legazpi Village)
        dropoff_loc_name : str
            The dropoff location name as a string, (e.g., Pine View, Legazpi Village)
        trip_distance : float
            The distance in meters
        fare_amount : float
            The amount paid by passenger
        pickup_loc_id : int
            A number assigned to the pickup location. This is based on the
            locations.csv file
        dropoff_loc_id : int
            A number assigned to the drop-off location. This is based on the
            locations.csv file

        Returns:
        ----------
        trip_ids: list
            Returns the list of trip index that are successfully added into the trips 
            database. Starting value of trip index is 0. 
            Prints a warning for trips that are already in the database or 
            has invalid or incomplete information.
        """

        trip_ids = []

        for i in range(len(list_of_trips)):
            list_current = list_of_trips[i]
            try:
                driver = list_current['driver']
                pickup_datetime = list_current['pickup_datetime']
                dropoff_datetime = list_current['dropoff_datetime']
                passenger_count = list_current['passenger_count']
                pickup_loc_name = list_current['pickup_loc_name']
                dropoff_loc_name = list_current['dropoff_loc_name']
                trip_distance = list_current['trip_distance']
                fare_amount = list_current['fare_amount']

            except:
                print(f'Warning: trip index {i} has invalid or incomplete information. '
                      'Skipping...')
            else:
                # initialize attributes
                self.driver = driver.strip()
                self.pickup_datetime = pickup_datetime
                self.dropoff_datetime = dropoff_datetime
                self.passenger_count = passenger_count
                self.trip_distance = trip_distance
                self.fare_amount = fare_amount

                # create the pickup_id and dropoff_id attributes
                df_locations = pd.read_csv(self.data_dir + '/locations.csv')
                filt_loc = (df_locations['loc_name'].str.lower()
                            == pickup_loc_name.lower().strip())
                self.pickup_id = int(
                    df_locations[filt_loc]['location_id'].values[0])
                filt_loc = (df_locations['loc_name'].str.lower()
                            == dropoff_loc_name.lower().strip())
                self.dropoff_id = int(
                    df_locations[filt_loc]['location_id'].values[0])

                # 1 trips_id
                try:
                    df_trip = pd.read_csv(self.data_dir + '/trips.csv')

                except:
                    trip_columns = np.array(['trip_id',
                                             'driver_id',
                                             'pickup_datetime',
                                             'dropoff_datetime',
                                             'passenger_count',
                                             'pickup_loc_id',
                                             'dropoff_loc_id',
                                             'trip_distance',
                                             'fare_amount'])
                    trip_id = 1
                    driver_id = 1
                    first_entry = np.array([trip_id,
                                            driver_id,
                                            self.pickup_datetime,
                                            self.dropoff_datetime,
                                            self.passenger_count,
                                            self.pickup_id,
                                            self.dropoff_id,
                                            self.trip_distance,
                                            self.fare_amount,
                                            ])
                    first_entry = first_entry.reshape(1, 9)
                    df_trip = pd.DataFrame(
                        data=first_entry, columns=trip_columns)
                    df_trip.to_csv(self.data_dir + '/trips.csv', index=False)

                else:
                    trip_id = len(df_trip.index) + 1
                    new_entry = [trip_id,
                                 self.driver,
                                 self.pickup_datetime,
                                 self.dropoff_datetime,
                                 self.passenger_count,
                                 self.pickup_id,
                                 self.dropoff_id,
                                 self.trip_distance,
                                 self.fare_amount,
                                 ]
                    df_driver = pd.read_csv(self.data_dir + '/drivers.csv')
                    last_name, given_name = self.driver.lower().split(', ')
                    filt = ((df_driver['given_name'].str.lower() == given_name)
                            & (df_driver['last_name'].str.lower() == last_name))

                    if df_driver[filt].empty:
                        new_entry[1] = len(df_driver.index) + 1

                    else:
                        new_entry[1] = int(df_driver[filt]['driver_id'])

                    df_trip_copy = df_trip.copy()
                    new_entry_copy = new_entry.copy()

                    if (df_trip_copy.iloc[:, 1:] == new_entry_copy[1:]).all(axis=1).any():
                        print(
                            f'Warning: trip index {i} is already in the database. Skipping...')
                        continue

                    else:
                        df_trip.loc[len(df_trip.index)] = new_entry
                        df_trip.to_csv(self.data_dir +
                                       '/trips.csv', index=False)

                # 2 driver_id
                try:
                    df_driver = pd.read_csv(self.data_dir + '/drivers.csv')

                except:
                    driver_columns = np.array(
                        ['driver_id', 'given_name', 'last_name'])
                    driver_id = 1
                    first_entry = np.array([driver_id,
                                            self.driver.split(', ')[1].title(),
                                            self.driver.split(', ')[0].title()]
                                           ).reshape(1, 3)
                    df_driver = pd.DataFrame(
                        data=first_entry, columns=driver_columns)
                    df_driver.to_csv(
                        self.data_dir + '/drivers.csv', index=False)

                else:
                    driver_id = len(df_driver.index) + 1
                    df_fullname = df_driver['last_name'] + \
                        ', ' + df_driver['given_name']

                    if self.driver.lower() not in [x.lower() for x in df_fullname.values.tolist()]:
                        new_entry = [driver_id,
                                     self.driver.split(', ')[1].title(),
                                     self.driver.split(', ')[0].title()
                                     ]
                        df_driver.loc[len(df_driver.index)] = new_entry
                        df_driver.to_csv(
                            self.data_dir + '/drivers.csv', index=False)

                trip_ids.append(trip_id)
        return trip_ids

    def delete_trip(self, *trip_id):
        """
        This method accepts a trip_id to delete then removes it from trips.csv. 
        This method will raise a SakayDBError if the trip_id is not found.
        """
        filepath = os.path.join(self.data_dir, 'trips.csv')
        if not os.path.exists(filepath):
            raise SakayDBError

        df = pd.read_csv(self.data_dir + '/trips.csv', index_col=False)
        if trip_id in df['trip_id'].values:
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df_updated = df.drop(
                df[df['trip_id'] == trip_id].index, inplace=True)
            df.to_csv(self.data_dir + '/trips.csv', index=False)

        else:
            raise SakayDBError

    def search_trips(self, **kwargs):
        """
        A method that accepts any of the keyword arguments with values passed 
        of any of following types:

        exact : for single value search. Some value with data type and format 
        conforming to that of key
        range : for range search

            Case 1: tuple like (value, None) sorts by key (chronological or 
                    ascending) returns all entries from value, begin inclusive
            Case 2: tuple like (None,value) sorts by key (chronological or 
                    ascending) returns all entries up to value, end inclusive
            Case 3: tuple like (value1, value2) sorts by key and returns values 
                    between value1 and value2, end inclusive.

        It raises sakayDBError when the following are not satisfied:

        1. Invalid keyword key i.e. not in listed keys above
        2. Invalid values for range i.e. tuples with sizes greater than 2, or 
            either values1 or value2 is not parsable as a datetime object.

        Parameters:

        driver_id (int): value assigned to the driver
        pickup_datetime (str): pickup date and time with format "hh:mm:ss,DD-MM-YYYY"
        dropoff_datetime (str): dropoff date and time with format "hh:mm:ss,DD-MM-YYYY"
        passenger_count (int): number of passengers 
        trip_distance (float): distance in meters 
        fare_amount (float): total fare amount 

        Returns:
        df: pd.DataFrame
            - DataFrame with all the entries aligned with search key and values

        """

        if len(kwargs) == 0:
            raise SakayDBError

        filepath = os.path.join(self.data_dir, 'trips.csv')
        if not os.path.exists(filepath):
            return []

        dt_format = '%H:%M:%S,%d-%m-%Y'
        def custom_date_parser(x): return datetime.strptime(x, dt_format)

        trips_df = pd.read_csv(self.data_dir + '/trips.csv',
                               parse_dates=['pickup_datetime',
                                            'dropoff_datetime'],
                               date_parser=custom_date_parser)

        trips_cols = ['driver_id', 'pickup_datetime', 'dropoff_datetime',
                      'passenger_count', 'trip_distance', 'fare_amount']
        trips_cols_dict = {'driver_id': int,
                           'fare_amount': float}

        for key, value in kwargs.items():

            len_input = np.size(np.array(value))

            if key == 'pickup_datetime' or key == 'dropoff_datetime':

                if (len_input == 2) and (value[0] is None) and (value[1] is None):
                    continue
                elif len_input == 1:
                    try:
                        value_single = datetime.strptime(value, dt_format)
                    except:
                        raise SakayDBError
                elif len_input == 2 and (value[0] is None):
                    try:
                        value_single = datetime.strptime(value[1], dt_format)
                    except:
                        raise SakayDBError
                elif len_input == 2 and (value[1] is None):
                    try:
                        value_single = datetime.strptime(value[0], dt_format)
                    except:
                        raise SakayDBError
                elif len_input == 2:
                    try:
                        value_0 = datetime.strptime(value[0], dt_format)
                        value_1 = datetime.strptime(value[1], dt_format)
                    except:
                        raise SakayDBError

            else:
                if (len_input == 2) and (value[0] is None) and (value[1] is None):
                    continue
                elif len_input == 1:
                    value_single = value
                elif len_input == 2 and (value[0] is None):
                    value_single = value[1]
                elif len_input == 2 and (value[1] is None):
                    value_single = value[0]
                elif len_input == 2:
                    value_0 = value[0]
                    value_1 = value[1]

            if key not in trips_cols:
                raise SakayDBError

            else:
                if len_input == 1:
                    filt = (trips_df[key] == value_single)
                    trips_df = trips_df[filt]

                elif len_input == 2 and (value[0] is None):
                    filt = (trips_df[key] <= value_single)
                    trips_df = trips_df[filt]

                elif len_input == 2 and (value[1] is None):
                    filt = (trips_df[key] >= value_single)
                    trips_df = trips_df[filt]

                elif len_input == 2:
                    filt = ((trips_df[key] >= value_0) &
                            (trips_df[key] <= value_1))
                    trips_df = trips_df[filt]
                else:
                    raise SakayDBError

        trips_df.iloc[:, 2] = trips_df.iloc[:, 2].apply(
            lambda x: x.strftime(dt_format))
        trips_df.iloc[:, 3] = trips_df.iloc[:, 3].apply(
            lambda x: x.strftime(dt_format))

        return trips_df

    def export_data(self):
        """
        A function that returns a dataframe with cumulative data from trips 
        sorted by trip ID and details the fundamental information from each 
        trip including even driver first names and surnames alongside pickup 
        and dropoff location names.

        Parameters:
        -----------
        given_name : str
            The trip driver's first name as a string.
        last_name : str
            The trip driver's surname as a string.
        pickup_datetime : str
            The pickup datetime as string with format "hh:mm:ss,DD-MM-YYYY".
        dropoff_datetime : str
            The dropoff datetime as string with format "hh:mm:ss,DD-MM-YYYY".
        passenger_count : int
            The number of passengers as integer.
        pickup_loc_name : str
            The pickup location name as a string, (e.g., Pine View, Legazpi Village).
        dropoff_loc_name : str
            The dropoff location name as a string, (e.g., Pine View, Legazpi Village).
        trip_distance : float
            The distance in meters.
        fare_amount : float
            The amount paid by passenger.
        """
        filepath = os.path.join(self.data_dir, 'trips.csv')
        if not os.path.exists(filepath):
            export_columns = ['driver_givenname', 'driver_lastname', 'pickup_datetime', 'dropoff_datetime',
                              'passenger_count', 'pickup_loc_name', 'dropoff_loc_name',
                              'trip_distance', 'fare_amount']
            return pd.DataFrame(data=None, columns=export_columns)

        self.df_driver = pd.read_csv(self.data_dir + '/drivers.csv')
        self.df_trip = pd.read_csv(self.data_dir + '/trips.csv')
        self.df_location = pd.read_csv(self.data_dir + '/locations.csv')
        self.df_trip['pickup_loc_id'] = self.df_trip['pickup_loc_id'].map(
            self.df_location.set_index('location_id')['loc_name'])
        self.df_trip['dropoff_loc_id'] = self.df_trip['dropoff_loc_id'].map(
            self.df_location.set_index('location_id')['loc_name'])
        self.df_trip.columns = self.df_trip.columns.str.replace(
            'pickup_loc_id', 'pickup_loc_name')
        self.df_trip.columns = self.df_trip.columns.str.replace(
            'dropoff_loc_id', 'dropoff_loc_name')
        self.df_trip.insert(2, "last_name", self.df_trip['driver_id'], True)
        self.df_trip['driver_id'] = self.df_trip['driver_id'].map(
            self.df_driver.set_index('driver_id')['given_name'])
        self.df_trip['last_name'] = self.df_trip['last_name'].map(
            self.df_driver.set_index('driver_id')['last_name'])
        self.df_trip.columns = self.df_trip.columns.str.replace(
            'driver_id', 'given_name')
        self.df_trip = self.df_trip.sort_values(by=['trip_id'], ascending=True)
        self.df_trip['driver_givenname'] = self.df_trip['given_name'].str.capitalize()
        self.df_trip['driver_lastname'] = self.df_trip['last_name'].str.capitalize()

        del self.df_trip['trip_id']
        del self.df_trip['given_name']
        del self.df_trip['last_name']

        return self.df_trip

    def generate_statistics(self, stat):
        """
        Create a method generate_statistics that returns a dictionary depending on the stat parameter passed to it:

        A method to return a a dictionary depending on stat paremeter based on the following:

        trip - returns Day name as key and average vehicle trips with pick-ups for that day
        passenger - key is unique passenger count while value is the a dictionary similar to trip
        driver - key is driver full name and value is a dictionary with day name as key and value
        average number of vehicle trips of driver on that day
        all - keys are trip, passenger and driver, values are the corresponding stat dictionaries returned by those keywords

        """
        self.stat = stat
        df_trips = self.export_data()

        df_trips["pickup_datetime"] = pd.to_datetime(df_trips["pickup_datetime"],
                                                     format="%H:%M:%S,%d-%m-%Y")
        df_trips["dropoff_datetime"] = pd.to_datetime(df_trips["dropoff_datetime"],
                                                      format="%H:%M:%S,%d-%m-%Y")
        df_trips["Date"] = df_trips["pickup_datetime"].dt.date
        df_trips["full_name"] = df_trips[["driver_lastname", "driver_givenname"]] \
            .apply(lambda x: ", ".join(x), axis=1)

        if self.stat == "trip":
            df_new = df_trips.groupby(
                by='Date')['fare_amount'].count().reset_index()
            df_new['Day'] = pd.to_datetime(df_new['Date']).dt.day_name()
            dict_new = df_new.groupby('Day')['fare_amount'].mean().to_dict()
            return dict_new

        elif self.stat == "passenger":
            pc = df_trips['passenger_count'].value_counts().sort_index().index
            count_dict = {}

            for count in pc:
                df_new = df_trips[df_trips['passenger_count'] == count]
                df_new = df_new.groupby(by='Date')[
                    'fare_amount'].count().reset_index()
                df_new['Day'] = pd.to_datetime(df_new['Date']).dt.day_name()
                dict_new = df_new.groupby(
                    'Day')['fare_amount'].mean().to_dict()
                count_dict[count] = dict_new
            return count_dict

        elif self.stat == "driver":
            driver_fullnames = df_trips["full_name"].value_counts().index
            driver_dict = {}

            for full_name in driver_fullnames:
                df_new = df_trips[df_trips['full_name'] == full_name]
                df_new = df_new.groupby(by='Date')[
                    'fare_amount'].count().reset_index()
                df_new['Day'] = pd.to_datetime(df_new['Date']).dt.day_name()
                dict_new = df_new.groupby(
                    'Day')['fare_amount'].mean().to_dict()
                driver_dict[full_name] = dict_new

            return driver_dict

        elif self.stat == "all":
            d1 = self.generate_statistics('trip')
            d2 = self.generate_statistics('passenger')
            d3 = self.generate_statistics('driver')
            return {"trip": d1, "passenger": d2, "driver": d3}

        else:
            raise SakayDBError

    def plot_statistics(self, stat):
        """
        A method that accept a stat parameter and returns a motplotlib.

        Parameters:
        ----------
        stat : str
            Specifies which plot to generate:
                For stat == trip, produce a vertical bar plot 

                For stat == passenger, produce a line plot with 
                multiple lines which represent a passenger count.

                For stat == driver, produce a 7x1 grid horizontal 
                bar plot 

        Returns:
        ----------
        stat : trip
            Returns a matplotlib Axes bar plot of the average number of
            vehicle trips per day of week. 

        stat : passenger
            produce a line plot with marker 'o' showing the average 
            number of vehicle trips per day for each of the different 
            passenger counts with multiple lines which represent a passenger 
            count.
        stat: driver
            produce a 7x1 grid horizontal 
            bar plot of the drivers with top average trips 
            per day. Each subplot correspond per day.
        """
        self.stat = stat
        day_index = ['Monday', 'Tuesday', 'Wednesday',
                     'Thursday', 'Friday', 'Saturday', 'Sunday']

        if self.stat == "trip":
            df = pd.DataFrame.from_dict(self.generate_statistics(
                "trip"), orient="index").reindex(day_index)
            bar_trip = df.plot.bar(xlabel="Day of week", ylabel="Ave Trips",
                                   title='Average trips per day', figsize=(12, 8))
            return bar_trip

        elif self.stat == "passenger":
            df = pd.DataFrame.from_dict(self.generate_statistics(
                "passenger"), orient="index").T.reindex(day_index)
            plot_pas = df.plot(xlabel="Day of week",
                               ylabel="Ave Trips", marker="o", figsize=(12, 8))
            return plot_pas

        elif self.stat == "driver":
            df = pd.DataFrame.from_dict(self.generate_statistics(
                "driver"), orient="index").T.reindex(day_index)
            df = df.T
            df1 = []
            df1.append(df["Monday"].sort_values().nlargest(
                5).sort_index(ascending=False).nsmallest(5))
            df1.append(df["Tuesday"].sort_values().nlargest(
                5).sort_index(ascending=False).nsmallest(5))
            df1.append(df["Wednesday"].sort_values().nlargest(
                5).sort_index(ascending=False).nsmallest(5))
            df1.append(df["Thursday"].sort_values().nlargest(
                5).sort_index(ascending=False).nsmallest(5))
            df1.append(df["Friday"].sort_values().nlargest(
                5).sort_index(ascending=False).nsmallest(5))
            df1.append(df["Saturday"].sort_values().nlargest(
                5).sort_index(ascending=False).nsmallest(5))
            df1.append(df["Sunday"].sort_values().nlargest(
                5).sort_index(ascending=False).nsmallest(5))
            fig, ax = plt.subplots(7, 1, figsize=(8, 25), sharex=True)

            for i in range(7):
                ax2 = df1[i].plot.barh(ax=ax[i], title="", legend=day_index[i])

            plt.xlabel("Ave Trips")
            return fig

        else:
            raise SakayDBError

    def _construct_od_matrix(self, df: pd.DataFrame):
        """
        A sub-method used in generate_od_matrix method. Returns the OD matrix for the 
        given DataFrame with its values as the average trips per day.

        Parameters:
        df : pd.DataFrame
            - DataFrame filtered by date_range in the generate_od_matrix method.

        Returns:
        od_matrix : pd.DataFrame
            - Origin-Destination matrix with values as the average trips per day.
        """
        _df = df.copy()
        _df['trip_count'] = 1
        _df['pickup_date'] = _df.pickup_datetime.str[-10:]
        return _df.groupby(['dropoff_loc_name', 'pickup_loc_name', 'pickup_date']).sum().unstack().trip_count.mean(axis=1).unstack().fillna(0)

    def generate_odmatrix(self, date_range=None):
        """
        Generates an OD matrix as a dataframe with pickup_datetime as the columns and
        dropoff_datetime as the rows. The value is the average daily trips of all
        origin-destination pairs within the specified date_range.

        Parameters:
        date_range : list-like, default = None
            - a list-like object of length 2 which specifies the date filter for the
                trips database.
        Returns:
        od_matrix : pd.DataFrame
            - Origin-Destination matrix with values as the average trips per day.
        """
        filepath = os.path.join(self.data_dir, 'trips.csv')
        if not os.path.exists(filepath):
            return pd.DataFrame({})
        else:
            self.trips_df = pd.read_csv(self.data_dir + '/trips.csv')
            self.locations_df = pd.read_csv(self.data_dir + '/locations.csv')
            trips_df = self.trips_df.copy()
            trips_df = trips_df.merge(self.locations_df, left_on='pickup_loc_id', right_on='location_id').merge(
                self.locations_df, left_on='dropoff_loc_id', right_on='location_id')
            trips_df.rename(columns={'loc_name_x': 'pickup_loc_name',
                            'loc_name_y': 'dropoff_loc_name'}, inplace=True)
            trips_df.drop(columns=['location_id_x',
                          'location_id_y'], inplace=True)
            if date_range is None:
                return self._construct_od_matrix(trips_df)
            try:
                if len(date_range) == 2:
                    date_range = pd.to_datetime(
                        date_range, format="%H:%M:%S,%d-%m-%Y")
                    trips_df['pickup_datetime_dt'] = pd.to_datetime(
                        trips_df.pickup_datetime, format="%H:%M:%S,%d-%m-%Y")
                    trips_df['dropoff_datetime_dt'] = pd.to_datetime(
                        trips_df.dropoff_datetime, format="%H:%M:%S,%d-%m-%Y")
                    if not date_range[0] is None and date_range[1] is None:
                        filtered_df = trips_df.sort_values(by='pickup_datetime')[
                            trips_df.pickup_datetime_dt >= date_range[0]]
                    elif not date_range[1] is None and date_range[0] is None:
                        filtered_df = trips_df.sort_values(by='pickup_datetime')[
                            trips_df.dropoff_datetime_dt <= date_range[1]]
                    else:
                        filtered_df = trips_df.sort_values(by='pickup_datetime')[
                            trips_df.pickup_datetime_dt >= date_range[0]][trips_df.dropoff_datetime_dt <= date_range[1]]
                    return self._construct_od_matrix(filtered_df)
            except:
                raise SakayDBError()
