from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

import scipy.sparse as sp

import random

# For automatic dataset downloading
from urllib2 import urlopen
from zipfile import ZipFile
from StringIO import StringIO
import shutil
import os.path


def data_iterator(data, batch_size):
    """
    A simple data iterator from https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
    :param data: list of numpy tensors that need to be randomly batched across their first dimension.
    :param batch_size: int, batch_size of data_iterator.
    Assumes same first dimension size of all numpy tensors.
    :return: iterator over batches of numpy tensors
    """
    # shuffle labels and features
    max_idx = len(data[0])
    idxs = np.arange(0, max_idx)
    np.random.shuffle(idxs)
    shuf_data = [dat[idxs] for dat in data]

    # Does not yield last remainder of size less than batch_size
    for i in range(max_idx//batch_size):
        data_batch = [dat[i*batch_size:(i+1)*batch_size] for dat in shuf_data]
        yield data_batch


def map_data(data):
    """
    Map data to proper indices in case they are not in a continues [0, N) range

    Parameters
    ----------
    data : np.int32 arrays

    Returns
    -------
    mapped_data : np.int32 arrays
    n : length of mapped_data

    """
    uniq = list(set(data))

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array(map(lambda x: id_dict[x], data))
    n = len(uniq)

    return data, id_dict, n


def download_dataset(dataset, files, data_dir):
    """ Downloads dataset if files are not present. """

    if not np.all([os.path.isfile(data_dir + f) for f in files]):
        url = "http://files.grouplens.org/datasets/movielens/" + dataset.replace('_', '-') + '.zip'
        request = urlopen(url)

        print('Downloading %s dataset' % dataset)
        if dataset in ['ml_100k', 'ml_1m']:
            target_dir = 'data/' + dataset.replace('_', '-')
        elif dataset == 'ml_10m':
            target_dir = 'data/' + 'ml-10M100K'
        else:
            raise ValueError('Invalid dataset option %s' % dataset)

        with ZipFile(StringIO(request.read())) as zip_ref:
            zip_ref.extractall('data/')

        source = [target_dir + '/' + s for s in os.listdir(target_dir)]
        destination = data_dir+'/'
        for f in source:
            shutil.copy(f, destination)

        shutil.rmtree(target_dir)


def load_data(fname, seed=1234, verbose=True):
    """ Loads dataset and creates adjacency matrix
    and feature matrix

    Parameters
    ----------
    fname : str, dataset
    seed: int, dataset shuffling seed
    verbose: to print out statements or not

    Returns
    -------
    num_users : int
        Number of users and items respectively

    num_items : int

    u_nodes : np.int32 arrays
        User indices

    v_nodes : np.int32 array
        item (movie) indices

    ratings : np.float32 array
        User/item ratings s.t. ratings[k] is the rating given by user u_nodes[k] to
        item v_nodes[k]. Note that that the all pairs u_nodes[k]/v_nodes[k] are unique, but
        not necessarily all u_nodes[k] or all v_nodes[k] separately.

    u_features: np.float32 array, or None
        If present in dataset, contains the features of the users.

    v_features: np.float32 array, or None
        If present in dataset, contains the features of the users.

    seed: int,
        For datashuffling seed with pythons own random.shuffle, as in CF-NADE.

    """

    u_features = None
    v_features = None

    print('Loading dataset', fname)

    data_dir = 'data/' + fname

    if fname == 'ml_100k':

        # Check if files exist and download otherwise
        files = ['/u.data', '/u.item', '/u.user']

        download_dataset(fname, files, data_dir)

        sep = '\t'
        filename = data_dir + files[0]

        dtypes = {
            'u_nodes': np.int32, 'v_nodes': np.int32,
            'ratings': np.float32, 'timestamp': np.float64}

        data = pd.read_csv(
            filename, sep=sep, header=None,
            names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.as_matrix().tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
        ratings = ratings.astype(np.float64)

        # Movie features (genres)
        sep = r'|'
        movie_file = data_dir + files[1]
        movie_headers = ['movie id', 'movie title', 'release date', 'video release date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                               names=movie_headers, engine='python')

        genre_headers = movie_df.columns.values[6:]
        num_genres = genre_headers.shape[0]

        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, g_vec in zip(movie_df['movie id'].values.tolist(), movie_df[genre_headers].values.tolist()):
            # Check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                v_features[v_dict[movie_id], :] = g_vec

        # User features

        sep = r'|'
        users_file = data_dir + files[2]
        users_headers = ['user id', 'age', 'gender', 'occupation', 'zip code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        occupation = set(users_df['occupation'].values.tolist())

        gender_dict = {'M': 0., 'F': 1.}
        occupation_dict = {f: i for i, f in enumerate(occupation, start=2)}

        num_feats = 2 + len(occupation_dict)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user id']
            if u_id in u_dict.keys():
                # age
                u_features[u_dict[u_id], 0] = row['age']
                # gender
                u_features[u_dict[u_id], 1] = gender_dict[row['gender']]
                # occupation
                u_features[u_dict[u_id], occupation_dict[row['occupation']]] = 1.

        u_features = sp.csr_matrix(u_features)
        v_features = sp.csr_matrix(v_features)

    elif fname == 'ml_1m':

        # Check if files exist and download otherwise
        files = ['/ratings.dat', '/movies.dat', '/users.dat']
        download_dataset(fname, files, data_dir)

        sep = r'\:\:'
        filename = data_dir + files[0]

        dtypes = {
            'u_nodes': np.int64, 'v_nodes': np.int64,
            'ratings': np.float32, 'timestamp': np.float64}

        # use engine='python' to ignore warning about switching to python backend when using regexp for sep
        data = pd.read_csv(filename, sep=sep, header=None,
                           names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], converters=dtypes, engine='python')

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.as_matrix().tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int64)
        ratings = ratings.astype(np.float32)

        # Load movie features
        movies_file = data_dir + files[1]

        movies_headers = ['movie_id', 'title', 'genre']
        movies_df = pd.read_csv(movies_file, sep=sep, header=None,
                                names=movies_headers, engine='python')

        # Extracting all genres
        genres = []
        for s in movies_df['genre'].values:
            genres.extend(s.split('|'))

        genres = list(set(genres))
        num_genres = len(genres)

        genres_dict = {g: idx for idx, g in enumerate(genres)}

        # Creating 0 or 1 valued features for all genres
        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, s in zip(movies_df['movie_id'].values.tolist(), movies_df['genre'].values.tolist()):
            # Check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                gen = s.split('|')
                for g in gen:
                    v_features[v_dict[movie_id], genres_dict[g]] = 1.

        # Load user features
        users_file = data_dir + files[2]
        users_headers = ['user_id', 'gender', 'age', 'occupation', 'zip-code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        # Extracting all features
        cols = users_df.columns.values[1:]

        cntr = 0
        feat_dicts = []
        for header in cols:
            d = dict()
            feats = np.unique(users_df[header].values).tolist()
            d.update({f: i for i, f in enumerate(feats, start=cntr)})
            feat_dicts.append(d)
            cntr += len(d)

        num_feats = sum(len(d) for d in feat_dicts)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user_id']
            if u_id in u_dict.keys():
                for k, header in enumerate(cols):
                    u_features[u_dict[u_id], feat_dicts[k][row[header]]] = 1.

        u_features = sp.csr_matrix(u_features)
        v_features = sp.csr_matrix(v_features)

    elif fname == 'ml_10m':

        # Check if files exist and download otherwise
        files = ['/ratings.dat']
        download_dataset(fname, files, data_dir)

        sep = r'\:\:'

        filename = data_dir + files[0]

        dtypes = {
            'u_nodes': np.int64, 'v_nodes': np.int64,
            'ratings': np.float32, 'timestamp': np.float64}

        # use engine='python' to ignore warning about switching to python backend when using regexp for sep
        data = pd.read_csv(filename, sep=sep, header=None,
                           names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], converters=dtypes, engine='python')

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.as_matrix().tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int64)
        ratings = ratings.astype(np.float32)

    else:
        raise ValueError('Dataset name not recognized: ' + fname)

    if verbose:
        print('Number of users = %d' % num_users)
        print('Number of items = %d' % num_items)
        print('Number of links = %d' % ratings.shape[0])
        print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

    return num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings, u_features, v_features
